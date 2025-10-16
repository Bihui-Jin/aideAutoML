from pathlib import Path
import shutil
import logging
import random
import time
from typing import Any, Callable, cast
import subprocess
import json

import os
import re

import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")

def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

def increase_timeout_limit(text, multiple: float = None, new_timeout: int = None):
    # Find the timeout value in parent_node.code using regex
    if new_timeout is not None:
        # Replace the timeout value in the template
        return re.sub(
            r'_timeout\s*=\s*\d+',
            f'_timeout = {new_timeout}',
            text
        )
    else:
        timeout_match = re.search(r'_timeout\s*=\s*(\d+)', text)
        if timeout_match:
            current_timeout = int(timeout_match.group(1))
            new_timeout = int(current_timeout * 1.5) if multiple is None else int(current_timeout * multiple)
            # Replace the timeout value in the template
            return (re.sub(
                r'_timeout\s*=\s*\d+',
                f'_timeout = {new_timeout}',
                text
            ), new_timeout)
        else:
            return (text, 30)
    
with open("/home/templates/draft_code_template.py", "r") as f:
    draft_code_template = f.read()
main_exec_match = re.search(
    r'# Main Execution Code Chunk.*?# ----------------------------\n(.*?)\n# ----------------------------\n# Finished',
    draft_code_template,
    re.DOTALL
)
main_exec_code_template = main_exec_match.group(1).strip()

ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        initial_code=None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self.initial_code = initial_code
        self.current_solution = None

    def search_policy2(self) -> Node | None:
        """
        Modified to return current solution instead of tree search.
        """
        return self.current_solution
    
    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "pyglove",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code should **implement the proposed solution** ",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program in PyGlove format that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3, qType=None) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None

        query_kwargs = {
            "system_message": prompt,
            "user_message": None,
            "model": self.acfg.code.model if qType != "_debug" else self.acfg.debug.model,
            "convert_system_to_user": self.acfg.convert_system_to_user,
        }

        if query_kwargs["model"] != "gpt-5":
            query_kwargs["temperature"] = self.acfg.code.temp

        # if self.acfg.code.model == "qwen3-max":
        #     query_kwargs["base_url"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

        logger.info(f"Model {query_kwargs['model']} is used for {qType}")

        for _ in range(retries):
            completion_text = query(**query_kwargs)

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        with open("/home/templates/draft_prompt.txt", "r") as f:
            draft_template = f.read()
        prompt["Instructions"] |= {
            "Symbolic Model Definition with Pyglove": draft_template,
            "Batching & tensors": [
                "Ensure all DataLoader batches are torch.Tensors (never Python lists) if DataLoader is used.",
                "Provide an explicit collate_fn in DataLoader that stacks items (inputs and targets) into tensors.",
                "Make targets float32 with a stable shape to avoid numpy scalars/0-D tensors.",
                "After collation, it must be valid to call .to(device) on both inputs and targets."
            ],
            "Solution sketch guideline": [
                # "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        prompt["Instructions"] |= {"Cautious while coding": [
            "Use the correct data paths: follow the structure shown in Data Overview.",
            "Keep the submission path exactly submission/submission.csv.",
            "Preserve test ID ordering: capture test IDs once and use that same ordering when writing predictions; do not re-read the test file at submission time.",
            "Ensure path handling is consistent (os.path.join with the nested directories).",
            "Be memory-safe between trials: free model/optimizer, del large tensors, call torch.cuda.empty_cache() if CUDA is used, and avoid OOMs.",
            "Validate file existence early and fail fast.",
            "Do not add new prints/logging.",
            ],
        }

        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        with open("/home/templates/draft_code_template.py", "r") as f:
            draft_code_template = f.read()
        prompt["Python Code Template"] = f"""
```
{draft_code_template}
```
"""

        plan, code = self.plan_and_code_query(prompt, qType="_draft")
        logger.info(f"Drafted code:\n{code}")
        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }

        timed_code, new_timeout = increase_timeout_limit(text=parent_node.code)
        prompt["Previous solution"] = {
            "Code": wrap_code(timed_code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        with open("/home/templates/improve_prompt.txt", "r") as f:
            improve_prompt = f.read()
        with open('/home/agent/output.txt', 'r') as output_file:
            output_perf = output_file.read()
        prompt["Instructions"] |= {
            "Solution requirments": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
            "Solution improvement sketch guideline": improve_prompt,
            "Is the higher the better": "True" if higher_better else "False",
            "Model performance": wrap_code(output_perf, lang=""),
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        with open("/home/templates/draft_code_template.py", "r") as f:
            draft_code_template = f.read()

        draft_code_template = increase_timeout_limit(draft_code_template, new_timeout=new_timeout)

        prompt["Python Code Template"] = f"""
```
{draft_code_template}
```
"""

        plan, code = self.plan_and_code_query(prompt, qType="_improve")
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should keep the PyGlove format and only fix bugs without introducing new bugs and changing the overall search space. "
            "Your response should not change the code architecture in `PyGlove` format and use `PyGlove` still. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        
        if os.path.exists('/home/agent/output.txt'):
            with open('/home/agent/output.txt', 'r') as output_file:
                output_perf = output_file.read()
        else:
            output_perf = ""

        timed_code, new_timeout = None, None
        # Due to timeout issues, we increase the timeout limit here
        if "Traceback (most recent call last):" not in parent_node.term_out and \
        parent_node.term_out.count("Trial failed with exception:") <10 and \
        ("Trial" not in output_perf or len([x for x in output_perf.split("\n") if "Trial" in x]) < 7):
            timed_code, new_timeout = increase_timeout_limit(text=parent_node.code)

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code) if timed_code is None else wrap_code(timed_code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        
        
        # Due to timeout issues, we increase the timeout limit here
        if "Traceback (most recent call last):" not in parent_node.term_out and \
        parent_node.term_out.count("Trial failed with exception:") <10 and \
        ("Trial" not in output_perf or len([x for x in output_perf.split("\n") if "Trial" in x]) < 7):
            prompt["Instructions"] |= {"Runtime Control": "Consider to add symbolic knobs to downsample the training set per trial (e.g., max_train_samples = pg.oneof(10000, 20000)), and in run() apply a deterministic (random_state) stratified subsample before the hold-out split; keep epochs small and prefer compact features (e.g., cap TF-IDF and use SVD) so each experiment finishes quickly."}

        prompt["Instructions"] |= self._prompt_resp_fmt

        if parent_node.term_out.count("pg.") > 1 or parent_node.term_out.count("pyglove") > 1:
            prompt["Instructions"] |= {
            "References of using Pyglove": 
"""- @pg.symbolize(*args, **kwargs): Make a symbolic class/function out of a regular Python class/function.  pg.symbolize is introduced for the purpose of making existing classes/functions symbolically programmable. For use cases that build symbolic classes from scratch (native PyGlove classes), extending pg.Object with @pg.members that declares the symbolic properties is the recommended way. pg.symbolize can be invoked as a class/function decorator, or as a function. When it is used as a decorator, the decorated class or function will be converted to a symbolic type.
    Parameters:
        *args 
            The positional arguments for symbolize are:
            class_or_fn: applicable when symbolize is called in function mode.
            constraints: an optional list of tuples that allows users to specify the constraints for arguments from the __init__ method (for class) or the arguments from the function signature (for function). Each tuple should be in format:
                (<arg_name>, <value_spec>, [description], [arg_metadata])
            Where arg_name is an argument name that is acceptable to the __init__ method of the class, or the function signature; 'value_spec' is a pg.ValueSpec object that validates the value of the argument. description and arg_metadata are optional, for documentation and meta-programming purposes.
        **kwargs 
            Keyword arguments will be passsed through to pg.wrap (for symbolizing classes) and pg.functor_class (for symbolizing functions).
    Returns:
        A Symbolic subclass for the decorated/input type.
    Examples:
        @pg.symbolize
        def foo(a, b):
        return a + b

        def foo(a, b):
        return a + b

        symbolic_foo = pg.symbolize(foo, [
            ('a', pg.typing.Int(min_value=0))
        ], returns=pg.typing.Int())

        class Foo:
        def __init__(self, a, b):
            self._a = a
            self._b = b
        def result(self):
            return self._a + self._b
        SymbolicFoo = pg.symbolize(Foo)

- @pg.members(fields, metadata=None, init_arg_list=None, serialization_key=None, additional_keys=None, add_to_registry=True): Function/Decorator for declaring symbolic fields for pg.Object.
    Parameters:
        fields: A list of pg.typing.Field or equivalent tuple representation as (<key>, <value-spec>, [description], [metadata-objects]). key should be a string. value-spec should be pg_typing.ValueSpec classes or equivalent, e.g. primitive values which will be converted to ValueSpec implementation according to its type and used as its default value. description is optional only when field overrides a field from its parent class. metadata-objects is an optional list of any type, which can be used to generate code according to the schema.
        metadata: Optional dict of user objects as class-level metadata which will be attached to class schema.
        init_arg_list: An optional sequence of strings as the positional argument list for __init__. This is helpful when symbolic attributes are inherited from base classes or the user want to change its order. If not provided, the init_arg_list will be automatically generated from symbolic attributes defined from pg.members in their declaration order, from the base classes to the subclass.
        serialization_key: An optional string to be used as the serialization key for the class during sym_jsonify. If None, cls.__type_name__ will be used. This is introduced for scenarios when we want to relocate a class, before the downstream can recognize the new location, we need the class to serialize it using previous key.
        additional_keys: An optional list of strings as additional keys to deserialize an object of the registered class. This can be useful when we need to relocate or rename the registered class while being able to load existing serialized JSON values.
        add_to_registry: If True, register serialization keys and additional keys with the class.
    Returns:
        a decorator function that register the class or function with schema created from the fields.
    Examples:
        @pg.members([
        # Declare symbolic fields. Each field produces a symbolic attribute
        # for its object, which can be accessed by `self.<field_name>`.
        # Description is optional.
        ('x', pg.typing.Int(min_value=0, default=0), 'Description for `x`.'),
        ('y', pg.typing.Str(), 'Description for `y`.')
        ])
        class A(pg.Object):
        def sum(self):
            return self.x + self.y

        @pg.members([
        # Override field 'x' inherited from class A and make it more restrictive.
        ('x', pg.typing.Int(max_value=10, default=5)),
        # Add field 'z'.
        ('z', pg.typing.Bool().noneable())
        ])
        class B(A):
        pass

        @pg.members([
        # Declare dynamic fields: any keyword can be acceptable during `__init__`
        # and can be accessed using `self.<field_name>`.
        (pg.typing.StrKey(), pg.typing.Int())
        ])
        class D(B):
        pass

        @pg.members([
        # Declare dynamic fields: keywords started with 'foo' is acceptable.
        (pg.typing.StrKey('foo.*'), pg.typing.Int())
        ])
        class E(pg.Object):
        pass
  
- pg.Object(*args, allow_partial=False, sealed=None, root_path=None, explicit_init=False, **kwargs): Base class for symbolic user classes. PyGlove allow symbolic programming interfaces to be easily added to most Python classes in two ways:
    1) Developing a dataclass-like symbolic class by subclassing pg.Object.
    2) Developing a class as usual and decorate it using pg.symbolize. This also work with existing classes.
    By directly subclassing pg.Object, programmers can create new symbolic classes with the least effort. For example:
        @pg.members([
            # Each tuple in the list defines a symbolic field for `__init__`.
            ('name', pg.typing.Str().noneable(), 'Name to greet'),
            ('time_of_day',
            pg.typing.Enum('morning', ['morning', 'afternnon', 'evening']),
            'Time of the day.')
        ])
        class Greeting(pg.Object):

        def __call__(self):
            # Values for symbolic fields can be accessed
            # as public data members of the symbolic object.
            print(f'Good {self.time_of_day}, {self.name}')

        # Create an object of Greeting and invoke it,
        # which shall print 'Good morning, Bob'.
        Greeting('Bob')()
    Symbolic fields can be inherited from the base symbolic class: the fields from the base class will be copied to the subclass in their declaration order, while the subclass can override the inherited fields with more restricted validation rules or different default values. For example:
        @pg.members([
            ('x', pg.typing.Int(max_value=10)),
            ('y', pg.typing.Float(min_value=0))
        ])
        class Foo(pg.Object)
        pass

        @pg.members([
            ('x', pg.typing.Int(min_value=1, default=1)),
            ('z', pg.typing.Str().noneable())
        ])
        class Bar(Foo)
        pass

        # Printing Bar's schema will show that there are 3 parameters defined:
        # x : pg.typing.Int(min_value=1, max_value=10, default=1))
        # y : pg.typing.Float(min_value=0)
        # z : pg.typing.Str().noneable()
        print(Bar.__schema__)

- pg.eq(left, right): Compares if two values are equal. Use symbolic equality if possible.
    Parameters:
        left: The left-hand value to compare.
        right: The right-hand value to compare.
    Returns:
        True if left and right is equal or symbolically equal. Otherwise False.

- pg.oneof(candidates, *, name=None, hints=None): N choose 1.
    Parameters:
        candidates: Candidates to select from. Items of candidate can be any type, therefore it can have nested hyper primitives, which forms a hierarchical search space.
        name: A name that can be used to identify a decision point in the search space. This is needed when the code to instantiate the same hyper primitive may be called multiple times under a pg.DynamicEvaluationContext.collect context or under a pg.DynamicEvaluationContext.apply context.
        hints: An optional value which acts as a hint for the controller.
    Returns:
        In symbolic mode, this function returns a ChoiceValue. In dynamic evaluation mode, this function returns one of the items in candidates. If evaluated under a pg.DynamicEvaluationContext.apply scope, this function will return the selected candidate. If evaluated under a pg.DynamicEvaluationContext.collect scope, it will return the first candidate.
    Example:
        @pg.members([
        ('x', pg.typing.Int())
        ])
        class A(pg.Object):
        pass

        # A single categorical choice:
        v = pg.oneof([1, 2, 3])

        # A complex type as candidate.
        v1 = pg.oneof(['a', {'x': 1}, A(1)])

        # A hierarchical categorical choice:
        v2 = pg.oneof([
            'foo',
            'bar',
            A(pg.oneof([1, 2, 3]))
        ])

- pg.floatv(min_value, max_value, scale=None, *, name=None, hints=None): A continuous value within a range.
    Parameters:
        min_value: Minimum acceptable value (inclusive).
        max_value: Maximum acceptable value (inclusive).
        scale: An optional string as the scale of the range. Supported values are None, 'linear', 'log', and 'rlog'. If None, the feasible space is unscaled. If linear, the feasible space is mapped to [0, 1] linearly. If log, the feasible space is mapped to [0, 1] logarithmically with 
            formula x -> log(x / min) / log(max / min).
            If rlog, the feasible space is mapped to [0, 1] “reverse” logarithmically, resulting in values close to max_value spread out more than the points near the min_value, with formula: x -> 1.0 - log((max + min - x) / min) / log (max / min). 
            min_value must be positive if scale is not None. Also, it depends on the search algorithm to decide whether this information is used or not.
        name: A name that can be used to identify a decision point in the search space. This is needed when the code to instantiate the same hyper primitive may be called multiple times under a pg.DynamicEvaluationContext.collect context or a pg.DynamicEvaluationContext.apply context.
        hints: An optional value which acts as a hint for the controller.
    Returns:
        In symbolic mode, this function returns a Float. In dynamic evaluate mode, this function returns a float value that is no less than the min_value and no greater than the max_value. If evaluated under an pg.DynamicEvaluationContext.apply scope, this function will return a chosen float value from the controller decisions. If evaluated under a pg.DynamicEvaluationContext.collect scope, it will return min_value.
    Example:
        # A continuous value within [0.0, 1.0]
        v = pg.floatv(0.0, 1.0)

- pg.manyof(k, candidates, distinct=True, sorted=False, *, name=None, hints=None, **kwargs): N choose K.
    Parameters:
        k: number of choices to make. Should be no larger than the length of candidates unless choice_distinct is set to False,
        candidates: Candidates to select from. Items of candidate can be any type, therefore it can have nested hyper primitives, which forms a hierarchical search space.
        distinct: If True, each choice needs to be unique.
        sorted: If True, choices are sorted by their indices in the candidates.
        name: A name that can be used to identify a decision point in the search space. This is needed when the code to instantiate the same hyper primitive may be called multiple times under a pg.DynamicEvaluationContext.collect context or a pg.DynamicEvaluationContext.apply context.
        hints: An optional value which acts as a hint for the controller.
        **kwargs: Keyword arguments for backward compatibility. choices_distinct: Old name for distinct. choices_sorted: Old name for sorted.
    Returns:
        In symbolic mode, this function returns a Choices. In dynamic evaluate mode, this function returns a list of items in candidates. If evaluated under a pg.DynamicEvaluationContext.apply scope, this function will return a list of selected candidates. If evaluated under a pg.DynamicEvaluationContext.collect scope, it will return a list of the first valid combination from the candidates. For example:

        # Evaluates to [0, 1, 2].
        manyof(3, range(5))
        # Evaluates to [0, 0, 0].
        manyof(3, range(5), distinct=False)
    Examples:
        @pg.members([
            ('x', pg.typing.Int())
        ])
        class A(pg.Object):
        pass

        # Chooses 2 distinct candidates.
        v = pg.manyof(2, [1, 2, 3])

        # Chooses 2 non-distinct candidates.
        v = pg.manyof(2, [1, 2, 3], distinct=False)

        # Chooses 2 distinct candidates sorted by their indices.
        v = pg.manyof(2, [1, 2, 3], sorted=True)

        # A complex type as candidate.
        v1 = pg.manyof(2, ['a', {'x': 1}, A(1)])

        # A hierarchical categorical choice:
        v2 = pg.manyof(2, [
            'foo',
            'bar',
            A(pg.oneof([1, 2, 3]))
        ])

- pg.typing.Union(candidates, default=MISSING_VALUE, is_noneable=False, frozen=False): Value spec for Union.
    Examples:
        # A required int or float value.
        pg.typing.Union([pg.typing.Int(), pg.typing.Float()])

        # An optional int or float value with default set to None.
        pg.typing.Union([pg.typing.Int(), pg.typing.Float()]).noneable()

        # A dict of specific keys, instance of class A or B, with {x=1} as its
        # default value.
        pg.typing.Union([
            pg.typing.Dict([
                ('x', pg.typing.Int(min_value=1)),
            ]),
            pg.typing.Object(A),
            pg.typing.Object(B),
        ], default={'x': 1})

- pg.typing.Floatdefault=MISSING_VALUE, min_value=None, max_value=None, is_noneable=False, frozen=False): Value spec for float type.
    Examples:
        # A required float value.
        pg.typing.Float()

        # A required float value with min and max value (both inclusive.)
        pg.typing.Float(min_value=1.0, max_value=10.0)

        # A float value with the default value set to 1
        pg.typing.Float(default=1)

        # An optional float value with default value set to None.
        pg.typing.Float().noneable()

        # An optional float value with default value set to 1.0.
        pg.typing.Float(default=1.0).noneable()

        # A frozen float with value set to 1.0 that is not modifiable by subclasses.
        pg.typing.Float().freeze(1)

- pg.typing.Int(default=MISSING_VALUE, min_value=None, max_value=None, is_noneable=False, frozen=False): Value spec for int type.
    Examples:
        # A required int value.
        pg.typing.Int()

        # A required int value with min and max value (both inclusive.)
        pg.typing.Int(min_value=1, max_value=10)

        # A int value with the default value set to 1
        pg.typing.Int(default=1)

        # An optional int value with default value set to None.
        pg.typing.Int().noneable()

        # An optional int value with default value set to 1.
        pg.typing.Int(default=1).noneable()

        # A frozen int with value set to 1 that is not modifiable by subclasses.
        pg.typing.Int().freeze(1)

- pg.typing.Str(default=MISSING_VALUE, regex=None, is_noneable=False, frozen=False): Value spec for string type.
    Examples:
        # A required str value.
        pg.typing.Str()

        # A required str value which matches with a regular expression.
        pg.typing.Str(regex='foo.*'))

        # A str value with the default value set to 'foo'.
        pg.typing.Str(default='foo')

        # An optional str value with default value set to None.
        pg.typing.Str().noneable()

        # An optional str value with default value set to 'foo'.
        pg.typing.Str(default='foo').noneable()

        # A frozen str with value set to 'foo' that is not modifiable by subclasses.
        pg.typing.Str().freeze('foo')

- pg.typing.Enum(default, values, frozen=False): Value spec for enum type.
    Examples:
        # A str enum value with options 'a', 'b', 'c' and its default set to 'a'.
        pg.typing.Enum('a', ['a', 'b', 'c'])

        # A mixed-type enum value.
        pg.typing.Enum('a', ['a', 5, True])

        # An optional enum value with default value set to 'a'.
        pg.typing.Enum('a', ['a', 'b', 'c']).noneable()

        # A frozen enum with value set to 'a' that is not modifiable by subclasses.
        pg.typing.Enum('a', ['a', 'b', 'c']).freeze('a')
""",}
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
                "Do not change the overall solution architecture and use PyGlove still.",
                "Do not change anything in the main execution code chunk, keeping it from the Main Execution Code Chunk Template below.",
            ],
            "Search budget & trials": [
                "Do not add or enforce timeouts; run trials to completion unless an actual error occurs.",
                "Do not introduce or depend on trial-count caps; assume any existing caps in the template are soft and unrelated to correctness."
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        
        prompt["Main Execution Code Chunk Template"] = f"""
```
{main_exec_code_template if new_timeout is None else increase_timeout_limit(main_exec_code_template, new_timeout=new_timeout)}
```
"""
        
        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt, qType="_debug")
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        logger.info(f"Debugged code:\n{code}")
        return new_node

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        """
        Linear single-solution iteration:
        - Use provided initial_code as the first solution (if present)
        - Evaluate it, then iteratively either debug or improve the current solution
        - Keep using existing evaluator (`parse_exec_result`), debug (`_debug`) and
          improve (`_improve`) helpers from this class so core features remain.
        """
        # clear the submission dir from previous steps
        shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
        (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        if self.initial_code is not None:
            # Use the given initial code for the first step
            if self.current_solution is None:
                code = extract_code(self.initial_code)
                nl_text = extract_text_up_to_code(self.initial_code)
                logger.info("Using provided initial code for the first step:\nPlan:{nl_text}\nCode:{code}".format(code=code, nl_text=nl_text))
                # Create initial node from provided code
                initial_node = Node(plan=nl_text, code=code)
                # Execute and evaluate the initial code immediately
                initial_node = self.parse_exec_result(
                            node=initial_node,
                            exec_result=exec_callback(initial_node.code, True),
                        )
                # Add to journal and set as current solution
                self.journal.append(initial_node)
                self.current_solution = initial_node
                logger.info(f"Using provided initial code solution, node type: {type(initial_node)}")

                # Handle the submission check for initial node
                if not initial_node.is_buggy:
                    if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                        initial_node.is_buggy = True
                        initial_node.metric = WorstMetricValue()
                        logger.info(f"Initial node {initial_node.id} did not produce a submission.csv")
                
                # Check if this initial node should be cached as best
                best_node = self.journal.get_best_node()
                if best_node is not None and best_node.id == initial_node.id:
                    logger.info(f"Initial node {initial_node.id} is the best node so far")
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_solution_dir.mkdir(exist_ok=True, parents=True)
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    best_submission_dir.mkdir(exist_ok=True, parents=True)
                    if (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                        shutil.copy(
                            self.cfg.workspace_dir / "submission" / "submission.csv",
                            best_submission_dir,
                        )
                    with open(best_solution_dir / "solution.py", "w") as f:
                        f.write(initial_node.code)
                    with open(best_solution_dir / "node_id.txt", "w") as f:
                        f.write(str(initial_node.id))
                
                self.current_step += 1
                return  # Exit early after processing initial code
        
            # Use current solution as parent for improvement/debugging
            parent_node = self.current_solution

            # Determine what action to take based on current solution state
            if parent_node.is_buggy:
                result_node = self._debug(parent_node)
            # else:
            #     result_node = self._improve(parent_node)

            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_callback(result_node.code, True),
            )

            # Update current solution to the new result
            self.current_solution = result_node
            
            # handle final cases where we missed buggy nodes somehow
            if not result_node.is_buggy:
                if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                    result_node.is_buggy = True
                    result_node.metric = WorstMetricValue()
                    logger.info(
                        f"Actually, node {result_node.id} did not produce a submission.csv"
                    )
            self.journal.append(result_node)

            # if the result_node is the best node, cache its submission.csv and solution.py
            # to best_solution/ by copying it there
            best_node = self.journal.get_best_node()
            if best_node is not None:
                if best_node.id == result_node.id:
                    logger.info(f"Node {result_node.id} is the best node so far")
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_solution_dir.mkdir(exist_ok=True, parents=True)
                    # copy submission/submission.csv to best_submission/submission.csv
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    best_submission_dir.mkdir(exist_ok=True, parents=True)
                    shutil.copy(
                        self.cfg.workspace_dir / "submission" / "submission.csv",
                        best_submission_dir,
                    )
                    # copy solution.py and relevant node id to best_solution/
                    with open(best_solution_dir / "solution.py", "w") as f:
                        f.write(result_node.code)
                    # take note of the node id of the best node
                    with open(best_solution_dir / "node_id.txt", "w") as f:
                        f.write(str(result_node.id))
                else:
                    logger.info(f"Node {result_node.id} is not the best node")
                    logger.info(f"Node {best_node.id} is still the best node")
            self.current_step += 1
        else:
            parent_node = self.search_policy()
            logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")

            if parent_node is None:
                result_node = self._draft()
            elif parent_node.is_buggy:
                result_node = self._debug(parent_node)
            else:
                result_node = self._improve(parent_node)

            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_callback(result_node.code, True),
            )
            # handle final cases where we missed buggy nodes somehow
            if not result_node.is_buggy:
                if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                    result_node.is_buggy = True
                    result_node.metric = WorstMetricValue()
                    logger.info(
                        f"Actually, node {result_node.id} did not produce a submission.csv"
                    )
            self.journal.append(result_node)

            # if the result_node is the best node, cache its submission.csv and solution.py
            # to best_solution/ by copying it there
            best_node = self.journal.get_best_node()
            if best_node is not None:
                if best_node.id == result_node.id:
                    logger.info(f"Node {result_node.id} is the best node so far")
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_solution_dir.mkdir(exist_ok=True, parents=True)
                    # copy submission/submission.csv to best_submission/submission.csv
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    best_submission_dir.mkdir(exist_ok=True, parents=True)
                    shutil.copy(
                        self.cfg.workspace_dir / "submission" / "submission.csv",
                        best_submission_dir,
                    )
                    # copy solution.py and relevant node id to best_solution/
                    with open(best_solution_dir / "solution.py", "w") as f:
                        f.write(result_node.code)
                    # take note of the node id of the best node
                    with open(best_solution_dir / "node_id.txt", "w") as f:
                        f.write(str(result_node.id))
                else:
                    logger.info(f"Node {result_node.id} is not the best node")
                    logger.info(f"Node {best_node.id} is still the best node")
            self.current_step += 1

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        if os.path.exists('/home/agent/output.txt'):
            with open('/home/agent/output.txt', 'r') as output_file:
                output_perf = output_file.read()
                logger.info(output_perf)

        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        query_kwargs = {
            "system_message": prompt,
            "user_message": None,
            "func_spec": review_func_spec,
            "model": self.acfg.feedback.model,
            "convert_system_to_user": self.acfg.convert_system_to_user,
        }

        if self.acfg.feedback.model != "gpt-5":
            query_kwargs["temperature"] = self.acfg.feedback.temp

        response = cast(
            dict,
            query(**query_kwargs),
            # query(
            #     **query_kwargs
            #     user_message=None,
            #     func_spec=review_func_spec,
            #     model=self.acfg.feedback.model,
            #     temperature=self.acfg.feedback.temp,
            #     convert_system_to_user=self.acfg.convert_system_to_user,
            # ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            (self.cfg.workspace_dir / "submission" / "submission.csv").exists() or
            (Path(".") / "submission" / "submission.csv").exists()
        )
        
        has_traceback = any(
            "Traceback (most recent call last):" in line
            for line in exec_result.term_out
        )

        # logger.info(f"Has CSV submission: {has_csv_submission}")
        # logger.info(f"response is_bug: {response['is_bug']}")
        # logger.info(f"response metric: {response['metric']}")
        # logger.info(f"response has_csv_submission: {response['has_csv_submission']}")
        # logger.info(f"script exec_result: {exec_result}")

        node.analysis = response["summary"]
        node.is_buggy = (
            (response["is_bug"] and has_traceback)
            or node.exc_type is not None
            or response["metric"] is None
            or response["has_csv_submission"] == False
            or has_csv_submission == False
        )

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            file_path = str(self.cfg.workspace_dir / "submission" / "submission.csv")
            res = subprocess.run(
                ["curl", "-sS", "-X", "POST",
                "-F", f"file=@{file_path}",
                'http://localhost:5000/grade'],
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(f"Grading result: {res.stdout}")

            grade = WorstMetricValue()
            if res.returncode == 0 and res.stdout is not None:
                try:
                    response_data = json.loads(res.stdout)
                    grade = response_data.get('score', WorstMetricValue())
                    if grade is None or not isinstance(grade, (float)):
                        grade = WorstMetricValue()
                    else:
                        grade = float(grade)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in grading response: {res.stdout}")
                    grade = WorstMetricValue()
            
            compare = {'jigsaw-toxic-comment-classification-challenge': False,
                        'google-quest-challenge': False,
                        'detecting-insults-in-social-commentary': False,
                        'tabular-playground-series-may-2022': False,
                        'denoising-dirty-documents': True,
                        'aerial-cactus-identification': False,
                        'tweet-sentiment-extraction': False,
                        'cassava-leaf-disease-classification': False,
                        'aptos2019-blindness-detection': False,
                        'random-acts-of-pizza': False,
                        'new-york-city-taxi-fare-prediction': True,
                        'nomad2018-predict-transparent-conductors': True,
                        'spooky-author-identification': True,
                        'mlsp-2013-birds': False,
                        'plant-pathology-2020-fgvc7': False,
                        'champs-scalar-coupling': True,
                        'uw-madison-gi-tract-image-segmentation': False,
                        'histopathologic-cancer-detection': False,
                        'bms-molecular-translation': True,
                        'predict-volcanic-eruptions-ingv-oe': True,
                        'h-and-m-personalized-fashion-recommendations': False,
                        'smartphone-decimeter-2022': True,
                        'hubmap-kidney-segmentation': False,
                        'whale-categorization-playground': False,
                        'text-normalization-challenge-russian-language': False,
                        'nfl-player-contact-detection': False,
                        'hms-harmful-brain-activity-classification': True,
                        'tensorflow2-question-answering': False,
                        'osic-pulmonary-fibrosis-progression': False,
                        'plant-pathology-2021-fgvc8': False,
                        'alaska2-image-steganalysis': False,
                        'hotel-id-2021-fgvc8': False,
                        'multi-modal-gesture-recognition': True,
                        'herbarium-2020-fgvc7': False,
                        'vesuvius-challenge-ink-detection': False,
                        '3d-object-detection-for-autonomous-vehicles': False,
                        'tabular-playground-series-dec-2021': False,
                        'inaturalist-2019-fgvc6': True,
                        'iwildcam-2020-fgvc7': False,
                        'seti-breakthrough-listen': False,
                        'icecube-neutrinos-in-deep-ice': True,
                        'herbarium-2022-fgvc9': False,
                        'herbarium-2021-fgvc8': False,
                        'vinbigdata-chest-xray-abnormalities-detection': False,
                        'rsna-breast-cancer-detection': False,
                        'us-patent-phrase-to-phrase-matching': False,
                        'chaii-hindi-and-tamil-question-answering': False,
                        'leaf-classification': True,
                        'statoil-iceberg-classifier-challenge': True,
                        'tgs-salt-identification-challenge': False,
                        'dog-breed-identification': True,
                        'lmsys-chatbot-arena': True,
                        'learning-agency-lab-automated-essay-scoring-2': False,
                        'ventilator-pressure-prediction': True,
                        'dogs-vs-cats-redux-kernels-edition': True,
                        'facebook-recruiting-iii-keyword-extraction': False,
                        'jigsaw-unintended-bias-in-toxicity-classification': False,
                        'ranzcr-clip-catheter-line-classification': False,
                        'text-normalization-challenge-english-language': False,
                        'billion-word-imputation': True,
                        'freesound-audio-tagging-2019': False,
                        'the-icml-2013-whale-challenge-right-whale-redux': False,
                        'petfinder-pawpularity-score': True,
                        'kuzushiji-recognition': False,
                        'iwildcam-2019-fgvc6': False,
                        'imet-2020-fgvc7': False,
                        'siim-isic-melanoma-classification': False,
                        'rsna-miccai-brain-tumor-radiogenomic-classification': False,
                        'siim-covid19-detection': False,
                        'rsna-2022-cervical-spine-fracture-detection': True,
                        'google-research-identify-contrails-reduce-global-warming': False,
                        'stanford-covid-vaccine': True,
                        'tensorflow-speech-recognition-challenge': False,
                        'AI4Code': False,
                        'cdiscount-image-classification-challenge': False}
            
            competition = os.getenv("COMPETITION_ID") 
            if competition in compare:
                maximize_setting = compare[competition]
            else:
                if response["lower_is_better"] is not None:
                    maximize_setting = not response["lower_is_better"]
                    compare[competition] = maximize_setting

            global higher_better
            higher_better = maximize_setting
            logger.info(f"Submission Grading: {grade}, Has csv: {has_csv_submission}, Is maximize: {maximize_setting}")
            
            if isinstance(grade, WorstMetricValue):
                node.metric = grade
            else:
                node.metric = MetricValue(
                    grade, maximize=not maximize_setting
                )

        return node

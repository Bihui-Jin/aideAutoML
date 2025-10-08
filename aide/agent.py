from pathlib import Path
import shutil
import logging
import random
import time
from typing import Any, Callable, cast
import subprocess
import json

import os

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
            "The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
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
        prompt["Instructions"] |= {
#             "Symbolic Model Definition with Pyglove": [
#                 "You MUST define the model as a **symbolic search space** using the `pyglove` library, not as a fixed architecture.",
#                 "The model must be a **neural network built with Torch layers**.",
#                 "Import pyglove as `import pyglove as pg`.",
#                 "Define the model architecture inside a class decorated with `@pg.symbolize`.",
#                 "Use PyGlove primitives (`pg.oneof`, `pg.manyof`, `pg.floatv`, `pg.intv`) to expose architectural knobs.",
#                 "Every symbolic choice MUST have a default value so the model is immediately runnable without search.",
#                 "For vision tasks: define a CNN backbone where choices include number of convolutional blocks, channel widths, kernel sizes, activation functions, and whether to add squeeze-excite layers.",
#                 "For sequence/NLP tasks: define a Transformer backbone where choices include number of layers, hidden size, number of attention heads, feed-forward expansion factor, and dropout rate.",
#                 "The symbolic draft must instantiate a runnable default model when called (e.g., `model = SymbolicCNN()` or `model = SymbolicTransformer()`).",
#                 "This draft model will later be modified by a NAS algorithm to explore the search space automatically.",
#                 "Example of a symbolic CNN block using Torch and PyGlove:",
#                 """
# @pg.symbolize
# class ConvBlock(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=pg.oneof([3, 5], default=3),
#             padding=pg.oneof([1, 2], default=1)
#         )
#         self.activation = pg.oneof([
#             torch.nn.ReLU(),
#             torch.nn.GELU()
#         ], default=torch.nn.ReLU())

#     def forward(self, x):
#         return self.activation(self.conv(x))
#                 """,
#                 "Ensure the model class can be instantiated and run forward without NAS tuning, but all symbolic knobs are available for future exploration."
#             ],
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt, qType="_draft")
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
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt, qType="_improve")
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this without introducing new bugs and changing the overall search space. "
            "Your response should not change the code architecture in `pyglove` format and use `pyglove` still. "
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
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt, qType="_debug")
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
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

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
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

            logger.info(f"Submission Grading: {grade}, Has csv: {has_csv_submission}, Is maximize: {maximize_setting}")
            
            if isinstance(grade, WorstMetricValue):
                node.metric = grade
            else:
                node.metric = MetricValue(
                    grade, maximize=not maximize_setting
                )

        return node

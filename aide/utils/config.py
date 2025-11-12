"""configuration and setup utils"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, cast

import sys, ast
from typing import Any

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from aide.journal import Journal, filter_journal

from . import tree_export
from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("aide")


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int

@dataclass
class RouletteModelConfig:
    model: str
    weight: float

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool

    code: StageConfig
    feedback: StageConfig
    debug: StageConfig

    search: SearchConfig

    max_no_improvement: int
    roulette_enabled: bool
    roulette_models: list[RouletteModelConfig]

@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    agent: AgentConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1

def _expand_roulette_models_arg(arg: str) -> list[str]:
    """
    Convert a single argument like:
      agent.roulette_models=[{'model': 'gpt-5', 'weight': 1}, {'model': 'claude', 'weight': 2}]
    into dotlist entries usable by OmegaConf:
      agent.roulette_models[0].model=gpt-5
      agent.roulette_models[0].weight=1
      agent.roulette_models[1].model=claude
      agent.roulette_models[1].weight=2
    """
    key, raw = arg.split("=", 1)
    # Strip surrounding whitespace
    raw = raw.strip()
    # Try ast.literal_eval first (handles Python list/dict repr)
    parsed: Any = None
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        # Fallback: attempt to coerce to JSON
        try:
            fixed = raw.replace("'", '"')
            parsed = json.loads(fixed)
        except Exception as e:
            raise ValueError(f"Failed to parse {key} value: {raw} ({e})")
    if not isinstance(parsed, list):
        raise ValueError(f"{key} must be a list, got {type(parsed)}")
    dotlist: list[str] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict) or "model" not in item or "weight" not in item:
            raise ValueError(f"Each roulette model must be a dict with model & weight. Got: {item}")
        dotlist.append(f"{key}[{i}].model={item['model']}")
        dotlist.append(f"{key}[{i}].weight={item['weight']}")
    return dotlist

def _expand_roulette_models_compact(raw: str) -> list[str]:
    """
    Parse compact form:
      gpt-5:1,claude-sonnet-4-5:1
    → dotlist entries.
    """
    dotlist = []
    if not raw:
        return dotlist
    for i, pair in enumerate(raw.split(",")):
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid roulette_models pair: {pair}")
        model, weight = pair.split(":", 1)
        dotlist.append(f"agent.roulette_models[{i}].model={model}")
        dotlist.append(f"agent.roulette_models[{i}].weight={weight}")
    return dotlist

def _collect_roulette_models_tokens(start_token: str, rest: list[str]) -> tuple[str, int]:
    """
    Reassemble split tokens for python-list form:
    agent.roulette_models=[{'model': 'gpt-5', 'weight': 1}, {'model': 'claude', 'weight': 1}]
    Returns joined string and number of extra tokens consumed.
    """
    buf = [start_token]
    consumed = 0
    if start_token.rstrip().endswith("]"):
        return start_token, consumed
    for t in rest:
        buf.append(t)
        consumed += 1
        if t.rstrip().endswith("]"):
            break
    return " ".join(buf), consumed

def _process_cli_args(argv: list[str]) -> list[str]:
    processed: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("agent.roulette_models="):
            raw_value = a.split("=", 1)[1]
            # Compact form?
            if ":" in raw_value and "[" not in raw_value:
                try:
                    processed.extend(_expand_roulette_models_compact(raw_value))
                except Exception as e:
                    print(f"[config] Warning (compact) roulette_models parse failed: {e}")
            else:
                # Possibly split python-list form; reassemble
                joined, consumed = _collect_roulette_models_tokens(a, argv[i+1:])
                try:
                    expanded = _expand_roulette_models_arg(joined)
                    processed.extend(expanded)
                except Exception as e:
                    print(f"[config] Warning: could not parse roulette models '{joined}': {e}")
                i += consumed
        else:
            processed.append(a)
        i += 1
    return processed

def _normalize_roulette_models(cfg):
    """
    Convert compact string or list into list[dict] with model, weight.
    Accepts formats:
      "gpt-5:1,claude-sonnet-4-5:1"
      [{'model': 'gpt-5', 'weight': 1}, {'model': 'claude-sonnet-4-5', 'weight': 1}]
    """
    try:
        rm = cfg.agent.get("roulette_models", None)
    except AttributeError:
        return

    if rm is None:
        return

    # Already structured (list of dicts) → leave
    if isinstance(rm, list):
        ok = all(isinstance(x, dict) and "model" in x and "weight" in x for x in rm)
        if ok:
            return

    # Compact string form
    if isinstance(rm, str):
        items = []
        for token in rm.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                print(f"[config] Skipping invalid roulette token: {token}")
                continue
            model, weight = token.split(":", 1)
            model = model.strip()
            try:
                weight_f = float(weight)
            except ValueError:
                print(f"[config] Invalid weight in roulette token: {token}")
                continue
            items.append({"model": model, "weight": weight_f})
        cfg.agent.roulette_models = items
        return

    # Python literal string (rare) fallback
    if isinstance(rm, str) and rm.startswith("["):
        try:
            parsed = ast.literal_eval(rm)
            if isinstance(parsed, list):
                cfg.agent.roulette_models = parsed
        except Exception:
            pass
        
def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        raw_cli = sys.argv[1:]
        dotlist = _process_cli_args(raw_cli)
        cli_conf = OmegaConf.from_dotlist(dotlist)
        # cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
        cfg = OmegaConf.merge(cfg, cli_conf)
    
    # Normalize roulette before structured merge
    _normalize_roulette_models(cfg)

    # Structured validation & fill defaults
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.roulette_models is None:
        cfg.agent.roulette_models = []
    
    return cast(Config, cfg)
    # return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal: Journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    # only if the journal has nodes
    if len(journal) > 0:
        tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
    # concatenate logs
    with open(cfg.log_dir / "full_log.txt", "w") as f:
        f.write(
            concat_logs(
                cfg.log_dir / "aide.log",
                cfg.workspace_dir / "best_solution" / "node_id.txt",
                cfg.log_dir / "filtered_journal.json",
            )
        )


def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the AIDE run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return f"File not found."

"""configuration and setup utils"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, cast

import sys, ast
from typing import Any

import coolname
import rich
from omegaconf import OmegaConf, DictConfig, ListConfig
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

def _expand_python_list_arg(raw: str) -> list[dict]:
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        try:
            parsed = json.loads(raw.replace("'", '"'))
        except Exception as e:
            raise ValueError(f"Failed to parse roulette_models literal: {raw} ({e})")
    if not isinstance(parsed, list):
        raise ValueError("roulette_models must be a list")
    out = []
    for item in parsed:
        if not isinstance(item, dict) or "model" not in item or "weight" not in item:
            raise ValueError(f"Invalid roulette_models entry: {item}")
        out.append({"model": str(item["model"]), "weight": float(item["weight"])})
    return out

def _parse_roulette_value(val: str) -> list[dict]:
    # Compact form: model:weight,model:weight
    if "[" not in val and "{" not in val:
        models = []
        for token in val.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid roulette token '{token}', expected model:weight")
            m, w = token.split(":", 1)
            models.append({"model": m.strip(), "weight": float(w.strip())})
        return models
    # Python / JSON list form
    return _expand_python_list_arg(val)

def _process_cli(argv: list[str]) -> dict:
    overrides: dict[str, Any] = {}
    i = 0
    # Reassemble possible split tokens for python-list forms
    while i < len(argv):
        tok = argv[i]
        if "=" not in tok:
            i += 1
            continue
        key, val = tok.split("=", 1)
        if key == "agent.roulette_models":
            # If value seems incomplete (doesn't end with ]), collect until complete
            if ("[" in val and "]" not in val) or ("{" in val and "}" not in val):
                buf = [val]
                j = i + 1
                while j < len(argv) and "]" not in buf[-1]:
                    buf.append(argv[j])
                    j += 1
                val = " ".join(buf)
                i = j - 1
            try:
                overrides["agent.roulette_models"] = _parse_roulette_value(val)
            except Exception as e:
                print(f"[config] Warning: roulette_models parse failed ({e}); ignoring override.")
        else:
            overrides[key] = val
        i += 1
    return overrides

def _apply_overrides(cfg: DictConfig, overrides: dict):
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)

def _coerce_roulette(cfg: DictConfig):
    rm = cfg.agent.get("roulette_models", None)
    if rm in (None, [], ()):
        cfg.agent.roulette_models = []
        return
    # Already a list of dict/config objects
    cleaned = []
    for item in rm:
        if isinstance(item, (DictConfig, dict)):
            model = item.get("model")
            weight = item.get("weight")
            if model is None or weight is None:
                continue
            cleaned.append({"model": str(model), "weight": float(weight)})
    cfg.agent.roulette_models = cleaned
    
def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    base = OmegaConf.load(path)
    if not isinstance(base, DictConfig):
        raise TypeError(f"Base config at {path} must be a mapping, got {type(base)}")
    if use_cli_args:
        overrides = _process_cli(sys.argv[1:])
        if overrides:
            _apply_overrides(base, overrides)
    # Normalize roulette before struct schema
    if "agent" in base:
        _coerce_roulette(base)
    # Build structured schema then merge (base last so its values override defaults)
    schema = OmegaConf.structured(Config)
    merged = OmegaConf.merge(schema, base)
    # Final safety: ensure list type
    if merged.agent.roulette_models is None:
        merged.agent.roulette_models = []
    # Cast entries to dataclass objects
    rm_objs = []
    for d in merged.agent.roulette_models:
        if isinstance(d, (DictConfig, dict)) and "model" in d and "weight" in d:
            rm_objs.append(RouletteModelConfig(model=str(d["model"]), weight=float(d["weight"])))
    merged.agent.roulette_models = rm_objs
    return cast(Config, merged)


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")
    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError("Provide either `goal` or `desc_file`.")
    if isinstance(cfg.data_dir, str):
        cfg.data_dir = Path(cfg.data_dir)
    cfg.data_dir = Path(cfg.data_dir).resolve()
    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()
    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)
    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()
    # Re-validate
    schema = OmegaConf.structured(Config)
    cfg = cast(Config, OmegaConf.merge(schema, cfg))
    return cfg


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

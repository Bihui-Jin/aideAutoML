import atexit
import logging
import shutil
import sys
import traceback

from . import backend

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.markdown import Markdown
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg


class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def fmt_metric(node: Node) -> str:
        val = getattr(getattr(node, "metric", None), "value", None)
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "N/A"
    
    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            metric_str = fmt_metric(node)
            if node is best_node:
                s = f"[{style}green]● {metric_str} (best)"
            else:
                s = f"[{style}green]● {metric_str}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def fmt_metric(node: Node) -> str:
        val = getattr(getattr(node, "metric", None), "value", None)
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "N/A"
    
    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            metric_str = fmt_metric(node)
            if marker_str:
                s = f"{indent}● {metric_str} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {metric_str} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str


def run():
    try:
        cfg = load_cfg()
        log_format = "[%(asctime)s] %(levelname)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
        )
        # dont want info logs from httpx
        httpx_logger: logging.Logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.WARNING)

        logger = logging.getLogger("aide")
        # save logs to files as well, using same format
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

        # we'll have a normal log file and verbose log file. Only normal to console
        file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.addFilter(VerboseFilter())

        verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
        verbose_file_handler.setFormatter(logging.Formatter(log_format))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.addFilter(VerboseFilter())

        logger.addHandler(file_handler)
        logger.addHandler(verbose_file_handler)
        logger.addHandler(console_handler)

        logger.info(f'Starting run "{cfg.exp_name}"')

        task_desc = load_task_desc(cfg)
        task_desc_str = backend.compile_prompt_to_md(task_desc)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)

        def cleanup():
            if global_step == 0:
                shutil.rmtree(cfg.workspace_dir)

        atexit.register(cleanup)

        initial_code = None
        with open('/home/agent/init_code.txt', 'r') as f:
            initial_code = f.read()

        if len(initial_code.strip()) == 0:
            initial_code = None
            logger.info("No initial code provided")
            
        journal = Journal()
        agent = Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            initial_code=initial_code,
        )

        interpreter = Interpreter(
            cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
        )

        global_step = len(journal)
        prog = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        status = Status("[green]Generating code...")
        prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

        def exec_callback(*args, **kwargs):
            status.update("[magenta]Executing code...")
            res = interpreter.run(*args, **kwargs)
            status.update("[green]Generating code...")
            return res

        def generate_live():
            tree = journal_to_rich_tree(journal)
            prog.update(prog.task_ids[0], completed=global_step)

            file_paths = [
                f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
                f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
                f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
            ]
            left = Group(
                Panel(Text(task_desc_str.strip()), title="Task description"),
                prog,
                status,
            )
            right = tree
            wide = Group(*file_paths)

            return Panel(
                Group(
                    Padding(wide, (1, 1, 1, 1)),
                    Columns(
                        [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                        equal=True,
                    ),
                ),
                title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
                subtitle="Press [b]Ctrl+C[/b] to stop the run",
            )

        while global_step < cfg.agent.steps:
            try:
                agent.step(exec_callback=exec_callback)
                # on the last step, print the tree
                if global_step == cfg.agent.steps - 1:
                    logger.info(journal_to_string_tree(journal))
                save_run(cfg, journal)
                global_step = len(journal)
            except StopIteration as e:
                # Early stopping triggered by agent
                logger.info(f"Early stopping triggered: {str(e)}")
                logger.info(journal_to_string_tree(journal))
                save_run(cfg, journal)
                break
        interpreter.cleanup_session()

        logger.info("AIDE completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("AIDE interrupted by user")
        if 'interpreter' in locals():
            interpreter.cleanup_session()
        return 130
    except Exception as e:
        logger.error(f"AIDE failed with error: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if 'interpreter' in locals():
            interpreter.cleanup_session()
        return 1


if __name__ == "__main__":
    exit_code = run()
    # prevents the script from hanging for hours when aide exits early due to errors, disk space issues, or other problems
    sys.exit(exit_code)
    # run()

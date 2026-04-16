from src.base_module.base_cli import CLIParser
from .task_7 import Module2PreprocessorTask7


def _factory(args):
    return Module2PreprocessorTask7(seed=args.seed)


def _add_args(parser):
    parser.add_argument("--mode", required=True, choices=["init", "check", "dry-run"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution", default="")
    parser.set_defaults(func=_factory)


cli_parser = CLIParser(
    name="module_2.submodule_2.task_7",
    add_cli_args=_add_args,
)
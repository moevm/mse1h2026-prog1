from src.base_module.base_cli import CLIParser
from .task_4 import Module2PreprocessorTask4


def _factory(args):
    return Module2PreprocessorTask4(seed=args.seed)


def _add_args(parser):
    parser.add_argument("--mode", required=True, choices=["init", "check", "dry-run"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution", default="")
    parser.set_defaults(func=_factory)


cli_parser = CLIParser(
    name="module_2.submodule_2.task_4",
    add_cli_args=_add_args,
)
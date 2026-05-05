from src.base_module.base_cli import CLIParser
from .task_5 import Module2HeaderFileTask5


def _factory(args):
    return Module2HeaderFileTask5(seed=args.seed)


def _add_args(parser):
    parser.add_argument("--mode", required=True, choices=["init", "check", "dry-run"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution", default="")
    parser.set_defaults(func=_factory)


cli_parser = CLIParser(
    name="module_2.submodule_3.task_5",
    add_cli_args=_add_args,
)
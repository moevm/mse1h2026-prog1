from src.base_module.base_cli import CLIParser
from .task_3 import Module2BuildStagesTask3


def _factory(args):
    return Module2BuildStagesTask3(seed=args.seed)


def _add_args(parser):
    parser.add_argument("--mode", required=True, choices=["init", "check", "dry-run"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution", default="")
    parser.set_defaults(func=_factory)


cli_parser = CLIParser(
    name="module_2.submodule_1.task_3",
    add_cli_args=_add_args,
)
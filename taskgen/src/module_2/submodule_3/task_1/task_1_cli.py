from src.base_module.base_cli import CLIParser
from .task_1 import Module2_Submodule3_Task1


def _factory(args):
    return Module2_Submodule3_Task1(seed=args.seed)


def _add_args(parser):
    parser.add_argument("--mode", required=True, choices=["init", "check", "dry-run"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution", default="")
    parser.set_defaults(func=_factory)


Module2_Submodule3_Task1_CLIParser = CLIParser(
    name="module_2.submodule_3.task_1",
    add_cli_args=_add_args,
)
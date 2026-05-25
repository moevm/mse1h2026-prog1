import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .task_14 import Module_1_Submodule_2_task_14


def create_task_limits(args) -> Module_1_Submodule_2_task_14:
    return Module_1_Submodule_2_task_14(**get_common_cli_args(args))


def add_cli_args_limits(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_limits)


Module_1_Submodule_2_task_14_CLI_Parser = CLIParser(
    name="module_1.submodule_2.task_14",
    add_cli_args=add_cli_args_limits,
)
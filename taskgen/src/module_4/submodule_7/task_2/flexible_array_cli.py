import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .flexible_array_task import FlexibleArrayTask


def create_task_flexible_array(args) -> FlexibleArrayTask:
    return FlexibleArrayTask(**get_common_cli_args(args))


def add_cli_args_flexible_array(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_flexible_array)


FlexibleArrayCLIParser = CLIParser(
    name="flexible_array",
    add_cli_args=add_cli_args_flexible_array,
)
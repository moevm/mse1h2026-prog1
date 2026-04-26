import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .typedef_print_task import TypedefPrintTask


def create_task_typedef_print(args) -> TypedefPrintTask:
    return TypedefPrintTask(**get_common_cli_args(args))


def add_cli_args_typedef_print(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_typedef_print)


TypedefPrintCLIParser = CLIParser(
    name="typedef_print",
    add_cli_args=add_cli_args_typedef_print,
)
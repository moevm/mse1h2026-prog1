import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .struct_sizeof_task import StructSizeofTask


def create_task_struct_sizeof(args) -> StructSizeofTask:
    return StructSizeofTask(**get_common_cli_args(args))


def add_cli_args_struct_sizeof(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_struct_sizeof)


StructSizeofCLIParser = CLIParser(
    name="struct_sizeof",
    add_cli_args=add_cli_args_struct_sizeof,
)
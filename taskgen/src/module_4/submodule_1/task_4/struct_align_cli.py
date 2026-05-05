import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .struct_align_task import StructAlignTask


def create_task_struct_align(args) -> StructAlignTask:
    return StructAlignTask(**get_common_cli_args(args))


def add_cli_args_struct_align(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_struct_align)


StructAlignCLIParser = CLIParser(
    name="struct_align",
    add_cli_args=add_cli_args_struct_align,
)
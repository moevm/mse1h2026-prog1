import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .union_fill_task import UnionFillTask


def create_task_union_fill(args) -> UnionFillTask:
    return UnionFillTask(**get_common_cli_args(args))


def add_cli_args_union_fill(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_union_fill)


UnionFillCLIParser = CLIParser(
    name="union_fill",
    add_cli_args=add_cli_args_union_fill,
)
import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .fread_task import FReadTask


def create_task_fread(args) -> FReadTask:
    return FReadTask(**get_common_cli_args(args))


def add_cli_args_fread(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_fread)


FReadCLIParser = CLIParser(
    name="fread",
    add_cli_args=add_cli_args_fread,
)
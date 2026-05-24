import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .getenv_task import GetEnvTask


def create_task_getenv(args) -> GetEnvTask:
    return GetEnvTask(**get_common_cli_args(args))


def add_cli_args_getenv(parser: argparse.ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--solution", type=str, default="solution.c")
    parser.add_argument("-n", "--n-tests", type=int, default=1)
    parser.add_argument("-a", "--all-tests", action="store_true")
    parser.add_argument("--mode", type=str, choices=("init", "check", "dry-run"), default="init")
    parser.set_defaults(func=create_task_getenv)


GetEnvCLIParser = CLIParser(
    name="getenv",
    add_cli_args=add_cli_args_getenv,
)
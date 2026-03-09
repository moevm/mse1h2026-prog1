import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .test_1 import TestTask


def create_task_test(args) -> TestTask:
    task = TestTask(
        # custom args
        # Common args
        **get_common_cli_args(args),
    )
    return task


def add_cli_args_test(parser: argparse.ArgumentParser):
    add_common_cli_args(parser)
    parser.set_defaults(func=create_task_test)


TestCLIParser = CLIParser(
    name="test_task",
    add_cli_args=add_cli_args_test,
)

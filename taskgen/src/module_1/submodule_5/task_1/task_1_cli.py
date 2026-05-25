import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .task_1 import Module_1_Submodule_5_task_1


def create_task_test(args) -> Module_1_Submodule_5_task_1:
    task = Module_1_Submodule_5_task_1(
        **get_common_cli_args(args),
    )
    return task


def add_cli_args_test(parser: argparse.ArgumentParser):
    add_common_cli_args(parser)
    parser.set_defaults(func=create_task_test)


Module_1_Submodule_5_task_1_CLI_Parser = CLIParser(
    name="module_1.submodule_5.task_1",
    add_cli_args=add_cli_args_test,
)

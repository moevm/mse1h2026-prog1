import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .task_1 import Task1  

def create_task_1(args) -> Task1:
    task = Task1(
        **get_common_cli_args(args),
    )
    return task

def add_cli_args_task_1(parser: argparse.ArgumentParser):
    add_common_cli_args(parser)
    parser.set_defaults(func=create_task_1)

Task1CLIParser = CLIParser(
    name="module_3.submodule_1.task_1", 
    add_cli_args=add_cli_args_task_1,
)

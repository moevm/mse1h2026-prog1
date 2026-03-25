import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .task_5 import Module3_Submodule2_Task5  

def create_task_1(args) -> Module3_Submodule2_Task5:
    task = Module3_Submodule2_Task5(
        **get_common_cli_args(args),
    )
    return task

def add_cli_args_task_1(parser: argparse.ArgumentParser):
    add_common_cli_args(parser)
    parser.set_defaults(func=create_task_1)

Module3_Submodule2_Task5CLIParser = CLIParser(
    name="module_3.submodule_2.task_5", 
    add_cli_args=add_cli_args_task_1,
)
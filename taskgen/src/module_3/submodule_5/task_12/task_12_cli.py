import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .task_12 import Module3_Submodule5_Task12 

def create_task_1(args) -> Module3_Submodule5_Task12:
    task = Module3_Submodule5_Task12(
        **get_common_cli_args(args),
    )
    return task

def add_cli_args_task_1(parser: argparse.ArgumentParser):
    add_common_cli_args(parser)
    parser.set_defaults(func=create_task_1)

Module3_Submodule5_Task12CLIParser = CLIParser(
    name="module_3.submodule_5.task_12", 
    add_cli_args=add_cli_args_task_1,
)
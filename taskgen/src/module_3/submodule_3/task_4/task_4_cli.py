import argparse
from src.base_module.base_cli import CLIParser, add_common_cli_args, get_common_cli_args
from .task_4 import Module3_Submodule3_Task4  

def create_task_1(args) -> Module3_Submodule3_Task4:
    task = Module3_Submodule3_Task4(
        **get_common_cli_args(args),
    )
    return task

def add_cli_args_task_1(parser: argparse.ArgumentParser):
    add_common_cli_args(parser)
    parser.set_defaults(func=create_task_1)

Module3_Submodule3_Task4CLIParser = CLIParser(
    name="module_3.submodule_3.task_4", 
    add_cli_args=add_cli_args_task_1,
)
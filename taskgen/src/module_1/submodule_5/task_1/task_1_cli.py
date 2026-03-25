import argparse

from src.base_module.base_cli import CLIParser, get_common_cli_args
from .task_1 import Module_1_Submodule_5_task_1



def _add_args(parser):
    parser.add_argument("--mode", required=True, choices=["init", "check", "dry-run"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution", default="")
    parser.set_defaults(func=_factory)
    

cli_parser = CLIParser(
    name="module_1.submodule_5.task_1",
    add_cli_args=_add_args,
)

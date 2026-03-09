from .base_module import BaseTaskClass, CLIParser
from .module_2.submodule_1.task_1 import cli_parser as module_2_submodule_1_task_1_cli
import importlib
import os
import sys
import inspect

def __load_task_modules():
    cur_loc = os.path.dirname(__file__)
    package_name = __package__ or "src"

    for name in os.listdir(cur_loc):
        if name.startswith("module_") and os.path.isdir(os.path.join(cur_loc, name)):
            try:
                mod = importlib.import_module(f".{name}", package_name)
                
                for obj_name, obj in inspect.getmembers(mod):
                    if (isinstance(obj, (BaseTaskClass, CLIParser))):
                        
                        setattr(sys.modules[__name__], obj_name, obj)
            except Exception as e:
                print(f"Ошибка загрузки модуля {name}: {e}")

__load_task_modules()
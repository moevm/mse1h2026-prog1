from .base_module import BaseTaskClass
import importlib
import os
import sys
import inspect

def load_task_modules():
    cur_loc = os.path.dirname(__file)
    package_name = package or "src"

    for name in os.listdir(cur_loc):
        if name.startswith("module_") and os.path.isdir(os.path.join(cur_loc, name)):
            try:
                mod = importlib.import_module(f".{name}", package_name)
                for obj_name, obj in inspect.getmembers(mod):
                    if inspect.isclass(obj) and issubclass(obj, BaseTaskClass) and obj is not BaseTaskClass:
                        setattr(sys.modules[name], obj_name, obj)
            except ImportError:
                continue

__load_task_modules()
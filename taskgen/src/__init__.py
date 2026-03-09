from .base_module import BaseTaskClass, CLIParser
import importlib
import os
import sys
import pkgutil
import inspect

def __load_task_modules():
    package_name = __name__
    package = sys.modules[package_name]

    for modinfo in pkgutil.walk_packages(package.__path__, package_name + "."):
        modname = modinfo.name

        if ".module_" not in modname:
            continue

        try:
            mod = importlib.import_module(modname)

            for obj_name, obj in inspect.getmembers(mod):

                if isinstance(obj, CLIParser):
                    setattr(package, obj_name, obj)

                elif inspect.isclass(obj) and issubclass(obj, BaseTaskClass) and obj is not BaseTaskClass:
                    setattr(package, obj_name, obj)

        except Exception as e:
            print(f"Ошибка загрузки модуля {modname}: {e}")

__load_task_modules()
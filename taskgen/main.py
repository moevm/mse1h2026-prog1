import argparse
import sys
import inspect
import src
import os

def init_task(task: src.BaseTaskClass):
    print(task.init_task())


def check_task(task: src.BaseTaskClass, solfile: str, name: str):
    task.load_student_solution(solfile)
    passed, msg = task.check()
    print("Passed:", passed)
    print(msg)
    if passed:
        if "NO_TOKEN" not in os.environ:
            text = src.generate_answer_token(f"{name}_{task.seed}")
            print("Ваш проверочный token будет выведен ниже:")
            print(text)
        sys.exit(0)
    
    sys.exit(1) 


def dry_run_task(task: src.BaseTaskClass):
    task.generate_task()
    task._generate_tests()
    for i, test in enumerate(task.tests):
        print(f"TEST #{i+1}:\n\t{test}")
    sys.exit(1)

# вызывается в случае не CLI
def summon_module_task(args_list: list = None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    for _, cli_parser in inspect.getmembers(src, lambda obj: isinstance(obj, src.CLIParser)):
        task_parser = subparsers.add_parser(cli_parser.name)
        cli_parser.add_cli_args(task_parser)

    args = parser.parse_args(args_list)
    task = args.func(args)
    match args.mode:
        case "init": init_task(task)
        case "check": check_task(task, args.solution, sys.argv[1])
        case "dry-run": dry_run_task(task)
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
   
    for _, cli_parser in inspect.getmembers(src, lambda obj: isinstance(obj, src.CLIParser)):
        task_parser = subparsers.add_parser(cli_parser.name)
        cli_parser.add_cli_args(task_parser)

    args = parser.parse_args()
    task = args.func(args)
    match args.mode:
        case "init": init_task(task)
        case "check": check_task(task, args.solution, sys.argv[1])
        case "dry-run": dry_run_task(task)
 
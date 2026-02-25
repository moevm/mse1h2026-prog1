from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


class CompileCommandTask(BaseTaskClass):
    """
    Задание: В текущей директории находится файл <src_name>.c, который выводит
    на экран сообщение "Hello, world!". Какую команду нужно ввести для
    компиляции, чтобы исполняемый файл назывался <src_name>.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        rng = random.Random(self.seed)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        src_name = ''.join(rng.choices(letters, k=rng.randint(5, 8)))
        self.file_name = src_name

    def generate_task(self) -> str:
        return(
            f"В текущей директории находится файл {self.file_name}.c, "
            f"который выводит на экран сообщение \"Hello, world!\". "
            f"Какую команду нужно ввести для компиляции, чтобы исполняемый файл назывался {self.file_name}?"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = f"gcc {self.file_name}.c -o {self.file_name}"

        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(
                    output.strip(), self.file_name)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def _compare_default(self, output: str, file_name: str) -> bool:
        tokens = output.strip().split()

        if len(tokens) != 4:
            return False

        if tokens[0] != "gcc":
            return False

        if tokens[1] == (file_name + ".c") and tokens[2] == "-o" and tokens[3] == file_name:
            return True

        if tokens[3] == (file_name + ".c") and tokens[1] == "-o" and tokens[2] == file_name:
            return True

        return False
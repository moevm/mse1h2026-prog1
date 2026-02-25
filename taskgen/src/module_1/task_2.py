from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


class StartCommandTask(BaseTaskClass):
    """
    Задание: В текущей директории находится файл <src_name>.c, который выводит
    на экран сообщение "Hello, world!". Была выполнена комманда: gcc <src_name>.c.
    Какая команда должна быть выполнена для запуска полученного исполняемого файла?
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
            f"Была выполнена комманда: gcc {self.file_name}.c. "
            f"Какая команда должна быть выполнена для запуска полученного исполняемого файла?"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = "./a.out"

        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, expected: self._compare_default(
                    output.strip(), expected)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected
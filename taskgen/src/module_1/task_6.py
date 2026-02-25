from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


class CountArgvTask(BaseTaskClass):
    """
    Задание: Чему равен argc, если argv = [<массив>]?
    int main(int argc, char *argv[]);
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        elements = ["program_name"]
        letters = 'abcdefghijklmnopqrstuvwxyz'
        for i in range(10):
            rng = random.Random(i * self.seed)
            element = ''.join(rng.choices(letters, k=rng.randint(3, 5)))
            elements.append(element)
        self.elements = elements[:self.seed % 10 + 1]

    def generate_task(self) -> str:
        return(
            f"Задание: Чему равен argc, если argv = {self.elements}?\n"
            "int main(int argc, char *argv[]);"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = str(len(self.elements))

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
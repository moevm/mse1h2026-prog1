from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


class CountMainFunctionTask(BaseTaskClass):
    """
    Задание: Сколько функций main содержит одна единица трансляции?
    При существовании несольких ответов на вопрос, введите наименьший
    и наибольший через запятую без пробелов.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def generate_task(self) -> str:
        return(
            f"Сколько функций main содержит одна единица трансляции? "
            f"При существовании несольких ответов на вопрос, введите наименьший и наибольший через запятую без пробелов."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = "1"

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
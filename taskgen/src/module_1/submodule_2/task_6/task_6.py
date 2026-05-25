from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_value(seed: int) -> int:
    r = seed % 3
    if r == 0:
        return (seed // 3) % 65536 - 32768
    elif r == 1:
        return 32768 + ((seed // 3) % (2147483647 - 32768 + 1))
    else:
        return 2147483648 + ((seed // 3) % 1000000000) * 1000 + (seed % 1000)


def expected_answer(value: int) -> str:
    if -32768 <= value <= 32767:
        return "short"
    elif -2147483648 <= value <= 2147483647:
        return "int"
    else:
        return "long"


def compare_answer(output: str, expected: str) -> bool:
    return output.strip().lower() == expected.strip().lower()


class Module_1_Submodule_2_task_6(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = generate_value(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Дано число `value` = {self.value}. "
            "Определите, каким одним ключевым словом (short, int или long) "
            "нужно объявить переменную, чтобы она гарантированно вместила это значение "
            "и занимала при этом минимальный размер памяти."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = expected_answer(self.value)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: compare_answer(output, exp)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.student_solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self):
        try:
            self.generate_task()
            expected = expected_answer(self.value)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
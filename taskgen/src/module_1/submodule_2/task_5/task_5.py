from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const(seed: int) -> int:
    return seed % 60000 + 1000


def generate_mul(seed: int) -> int:
    return (seed // 60000) % 50 + 2


def compute_answer(const: int, mul: int) -> str:
    return str((const * mul) % 65536)


def compare_answer(output: str, expected: str) -> bool:
    normalized_output = output.strip()
    normalized_expected = expected.strip()
    return normalized_output == normalized_expected


class Module_1_Submodule_2_task_5(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const = generate_const(self.seed)
        self.mul = generate_mul(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        code = (
            f"unsigned short s = {self.const};\n"
            f"s = s * {self.mul};\n"
            'printf("%hu", s);'
        )
        return (
            "Ниже представлен фрагмент программы. Определите, что он выведет.\n\n"
            + code
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = compute_answer(self.const, self.mul)
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
            expected = compute_answer(self.const, self.mul)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
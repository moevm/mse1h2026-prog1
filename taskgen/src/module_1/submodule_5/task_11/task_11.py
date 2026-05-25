from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_A(seed: int) -> int:
    random.seed(seed)
    return random.randint(-2147483648, 2147483647)


def generate_S(seed: int) -> int:
    random.seed(seed + 1)
    return random.randint(0, 31)


def compute_output(A: int, S: int) -> str:
    a_shifted = A >> S
    b = A & 0xFFFFFFFF
    b_shifted = (b >> S) & 0xFFFFFFFF
    return f"{a_shifted} {b_shifted}"


class Module_1_Submodule_5_task_11(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = generate_A(self.seed)
        self.S = generate_S(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Ниже представлен фрагмент программы. Что он выведет? Запишите два числа через пробел.\n"
            f"signed int a = {self.A};\n"
            f"int s = {self.S};\n"
            f"unsigned int b = (unsigned int)a;\n"
            f"printf(\"%d %u\", a >> s, b >> s);\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = compute_output(self.A, self.S)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda out, exp: out.strip() == exp.strip()
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
            expected = compute_output(self.A, self.S)
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
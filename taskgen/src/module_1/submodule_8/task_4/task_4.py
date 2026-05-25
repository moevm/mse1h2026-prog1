from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_A(seed: int) -> int:
    random.seed(seed)
    return random.randint(32, 126)


def compute_output(A: int) -> str:
    hex_val = format(A, 'x')
    return f"{A} {chr(A)} {hex_val}"


class Module_1_Submodule_8_task_4(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = generate_A(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Что именно напечатает фрагмент программы? Запишите три символа (или числа) через пробел, которые появятся на экране.\n"
            f"int a = {self.A};\n"
            f"printf(\"%d %c %x\", a, a, a);\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = compute_output(self.A)
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
            expected = compute_output(self.A)
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
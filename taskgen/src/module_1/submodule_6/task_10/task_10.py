from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_M(seed: int) -> int:
    random.seed(seed)
    return random.randint(1, 20)


def generate_N(seed: int) -> int:
    random.seed(seed + 1)
    return random.randint(1, 20)


def generate_S(seed: int) -> int:
    random.seed(seed + 2)
    return random.randint(1, 40)


def compute_count(M: int, N: int, S: int) -> int:
    total = M * N
    excluded = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if i + j == S:
                excluded += 1
    return total - excluded


class Module_1_Submodule_6_task_10(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.M = generate_M(self.seed)
        self.N = generate_N(self.seed)
        self.S = generate_S(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Сколько чисел будет выведено на экран в результате работы следующей программы?\n"
            f"#include <stdio.h>\n\n"
            f"int main()\n"
            f"{{\n"
            f"    int count = 0;\n"
            f"    for (int i = 1; i <= {self.M}; i++)\n"
            f"    {{\n"
            f"        for (int j = 1; j <= {self.N}; j++)\n"
            f"        {{\n"
            f"            if (i + j == {self.S})\n"
            f"                continue;\n"
            f"            count++;\n"
            f"        }}\n"
            f"    }}\n"
            f"    printf(\"%d\", count);\n"
            f"    return 0;\n"
            f"}}\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = str(compute_count(self.M, self.N, self.S))
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
            expected = str(compute_count(self.M, self.N, self.S))
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
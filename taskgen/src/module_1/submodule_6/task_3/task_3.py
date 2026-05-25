from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def c_mod(a: int, m: int) -> int:
    return a - int(a / m) * m


def generate_const1(seed: int) -> int:
    random.seed(seed)
    return random.randint(-50, 50)


def generate_const2(seed: int) -> int:
    random.seed(seed + 1)
    return random.randint(-50, 50)


def generate_const3(seed: int) -> int:
    random.seed(seed + 2)
    return random.randint(-50, 50)


def compute_output(a: int, b: int, c: int) -> str:
    rem_a = c_mod(a, 3)
    if rem_a == 0:
        rem_b = c_mod(b, 2)
        if rem_b == 0:
            result = a + b + c
        elif rem_b == 1:
            result = a - b + c
        else:
            result = a + b + c
    elif rem_a == 1:
        if c > 5:
            result = a * b
        else:
            result = a + b
    elif rem_a == 2:
        rem_c = c_mod(c, 2)
        if rem_c == 0:
            result = a * c
        else:
            result = b * c
    else:
        result = a + b + c
    return str(result)


class Module_1_Submodule_6_task_3(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const1 = generate_const1(self.seed)
        self.const2 = generate_const2(self.seed)
        self.const3 = generate_const3(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Напишите, что выведет программа:\n"
            f"#include <stdio.h>\n\n"
            f"int main() {{\n"
            f"    int a = {self.const1};\n"
            f"    int b = {self.const2};\n"
            f"    int c = {self.const3};\n\n"
            f"    switch (a % 3) {{\n"
            f"        case 0:\n"
            f"            switch (b % 2) {{\n"
            f"                case 0:\n"
            f"                    printf(\"%d\\n\", a + b + c);\n"
            f"                    break;\n"
            f"                case 1:\n"
            f"                    printf(\"%d\\n\", a - b + c);\n"
            f"                    break;\n"
            f"            }}\n"
            f"            break;\n\n"
            f"        case 1:\n"
            f"            if (c > 5) {{\n"
            f"                printf(\"%d\\n\", a * b);\n"
            f"            }} else {{\n"
            f"                printf(\"%d\\n\", a + b);\n"
            f"            }}\n"
            f"            break;\n\n"
            f"        case 2:\n"
            f"            switch (c % 2) {{\n"
            f"                case 0:\n"
            f"                    printf(\"%d\\n\", a * c);\n"
            f"                    break;\n"
            f"                default:\n"
            f"                    printf(\"%d\\n\", b * c);\n"
            f"                    break;\n"
            f"            }}\n"
            f"            break;\n\n"
            f"        default:\n"
            f"            printf(\"%d\\n\", a + b + c);\n"
            f"    }}\n\n"
            f"    return 0;\n"
            f"}}\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = compute_output(self.const1, self.const2, self.const3)
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
            expected = compute_output(self.const1, self.const2, self.const3)
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
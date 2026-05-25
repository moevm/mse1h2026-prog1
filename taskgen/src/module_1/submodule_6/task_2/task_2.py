from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const1(seed: int) -> int:
    random.seed(seed)
    return random.randint(-100, 100)


def generate_const2(seed: int) -> int:
    random.seed(seed + 1)
    return random.randint(-100, 100)


def generate_const3(seed: int) -> int:
    random.seed(seed + 2)
    return random.randint(-100, 100)


def generate_bool(seed: int) -> int:
    random.seed(seed + 3)
    return random.randint(0, 1)


def compute_output(a: int, b: int, c: int, bool_val: int) -> str:
    if a % 2 == bool_val:
        if b // 7 > c:
            result = a + b
        else:
            result = a - b
    else:
        result = a + c - b
    return str(result)


class Module_1_Submodule_6_task_2(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const1 = generate_const1(self.seed)
        self.const2 = generate_const2(self.seed)
        self.const3 = generate_const3(self.seed)
        self.bool_val = generate_bool(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Напишите, что выведет программа:\n"
            f"#include <stdio.h>\n\n"
            f"int main()\n"
            f"{{\n"
            f"    int a = {self.const1};\n"
            f"    int b = {self.const2};\n"
            f"    int c = {self.const3};\n"
            f"    if (a % 2 == {self.bool_val}) \n"
            f"    {{\n"
            f"        if (b / 7 > c) \n"
            f"        {{\n"
            f"            printf(\"%d\\n\", a + b);\n"
            f"        }}\n"
            f"        else \n"
            f"        {{\n"
            f"            printf(\"%d\\n\", a - b);\n"
            f"        }}\n"
            f"    }}\n"
            f"    else \n"
            f"    {{\n"
            f"        printf(\"%d\\n\", a + c - b);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = compute_output(self.const1, self.const2, self.const3, self.bool_val)
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
            expected = compute_output(self.const1, self.const2, self.const3, self.bool_val)
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
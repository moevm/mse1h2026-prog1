from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const1(seed: int) -> int:
    return seed % 100 + 1


def generate_const2(seed: int) -> int:
    return (seed // 100) % 100 + 1


def check_answer(a: int, b: int) -> str:
    return f"{2 * a + 3 * b} {a + 2 * b}"


def compare_answer(output: str, expected: str) -> bool:
    normalized_output = " ".join(output.strip().split())
    normalized_expected = " ".join(expected.strip().split())
    return normalized_output == normalized_expected


class Module_1_Submodule_1_task_7(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const1 = generate_const1(self.seed)
        self.const2 = generate_const2(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        code = (
            "#include <stdio.h>\n\n"
            "int main()\n"
            "{\n"
            f"    int a = {self.const1};\n"
            f"    int b = {self.const2};\n"
            "    a = a + b;\n"
            "    b = a + b;\n"
            "    a = a + b;\n"
            '    printf("%d %d", a, b);\n'
            "    return 0;\n"
            "}"
        )
        return (
            "Дана программа. Определите, что она выведет на экран (два числа через пробел).\n\n"
            + code
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.const1, self.const2)
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
            expected = check_answer(self.const1, self.const2)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
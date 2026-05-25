from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const(seed: int) -> int:
    return seed % 256


def compute_answer(const: int) -> str:
    a_signed = (const * 2) & 0xFF
    if a_signed >= 128:
        a_signed -= 256
    b_unsigned = (const * 2) & 0xFF
    return f"{a_signed} {b_unsigned}"


def compare_answer(output: str, expected: str) -> bool:
    normalized_output = " ".join(output.strip().split())
    normalized_expected = " ".join(expected.strip().split())
    return normalized_output == normalized_expected


class Module_1_Submodule_2_task_4(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const = generate_const(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        code = (
            "#include <stdio.h>\n\n"
            "int main() {\n"
            f"    signed char a = {self.const};\n"
            f"    unsigned char b = {self.const};\n"
            "    a = a * 2;\n"
            "    b = b * 2;\n"
            '    printf("%d %d", (int)a, (int)b);\n'
            "    return 0;\n"
            "}"
        )
        return (
            "Определите, какие числа будут выведены на экран после выполнения следующей программы. "
            "Числа напишите через пробел.\n\n" + code
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = compute_answer(self.const)
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
            expected = compute_answer(self.const)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
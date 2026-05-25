from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_type(seed: int) -> str:
    types = ["char", "short", "int", "long", "long long"]
    return types[seed % len(types)]


def expected_bits(type_name: str) -> int:
    sizes = {
        "char": 8,
        "short": 16,
        "int": 32,
        "long": 32,
        "long long": 64
    }
    return sizes[type_name]


def compare_answer(output: str, expected: str) -> bool:
    return output.strip() == expected.strip()


class Module_1_Submodule_2_task_7(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = generate_type(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Определите, сколько бит занимает переменная типа `{self.type}` "
            "в среде gcc x86. В ответе укажите одно число."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = str(expected_bits(self.type))
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
            expected = str(expected_bits(self.type))
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
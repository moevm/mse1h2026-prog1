from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const(seed: int) -> int:
    return seed % 1000 + 1


def check_answer(const: int) -> str:
    return f"int main() {{ return {const}; }}"


def compare_answer(output: str, expected: str) -> bool:
    normalized_output = " ".join(output.strip().split())
    variants = [" ".join(item.strip().split()) for item in expected.split("||")]
    return normalized_output in variants


class Module_1_Submodule_1_task_6(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const = generate_const(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Напишите функцию main. Она должна возвращать {self.const} "
            f"и больше ничего делать не должна.\n\n"
            f"Ввод: пуст.\n\n"
            f"Вывод:\n"
            f"int main()\n{{\n    return {self.const};\n}}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.const)
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
            expected = check_answer(self.const)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
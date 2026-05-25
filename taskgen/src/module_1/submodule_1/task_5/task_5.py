from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_consts(seed: int) -> tuple[int, float, str]:
    rng = random.Random(seed)
    const1 = rng.randint(1, 99)
    const2 = rng.randint(1, 99) + rng.randint(0, 99) / 100
    const3 = chr(ord("A") + (seed % 26))
    return const1, const2, const3


def check_answer(const1: int, const2: float, const3: str) -> str:
    return f".x={const1}, y ={const2:.6f} z= {const3}"


class Module_1_Submodule_1_task_5(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const1, self.const2, self.const3 = generate_consts(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            "Что выведет программа при запуске?\n"
            "#include <stdio.h>\n\n"
            "int main()\n"
            "{\n"
            f"    int x = {self.const1};\n"
            f"    double y = {self.const2:.2f};\n"
            f"    char z = '{self.const3}';\n"
            "    printf(\".x=%d, y =%f z= %c\\n\", x, y, z);\n"
            "    return 0;\n"
            "}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.const1, self.const2, self.const3)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(output.strip(), exp)
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
            expected = check_answer(self.const1, self.const2, self.const3)
            student = self.student_solution.strip()
            if self._compare_default(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
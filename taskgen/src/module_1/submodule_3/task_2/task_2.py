from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_number_expr(seed: int) -> tuple[str, str]:
    variant = seed % 4
    s = seed // 4
    a = (s % 9) + 1
    s //= 9
    b = s % 5
    s //= 5
    c = (s % 9) + 1
    s //= 9
    d = s % 5

    if variant == 0:
        expr = f"{a}e{b}"
        value = a * (10 ** b)
    elif variant == 1:
        expr = f"{a}e{b} + {c}e{d}"
        value = a * (10 ** b) + c * (10 ** d)
    elif variant == 2:
        expr = f"{a}e{b} - {c}e{d}"
        value = a * (10 ** b) - c * (10 ** d)
    else:
        expr = f"{c}e{d} + {a}e{b}"
        value = c * (10 ** d) + a * (10 ** b)

    return expr, str(int(value))


class Module_1_Submodule_3_task_2(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_expr, self.expected = generate_number_expr(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Напишите, как выглядит число `{self.number_expr}` в привычной форме. "
            f"Например: `1e2 = 100`."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=self.expected,
                compare_func=lambda output, exp: output.strip() == exp.strip()
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
            student = self.student_solution.strip()
            if student == self.expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
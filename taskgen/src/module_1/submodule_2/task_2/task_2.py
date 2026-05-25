from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


SPECS = ["%c", "%d", "%x", "%o", "%b"]


def generate_params(seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    letter = chr(ord("a") + rng.randint(0, 25))
    number = rng.randint(1, 20)
    op = "+" if rng.randint(0, 1) == 0 else "-"
    expression = f"'{letter}' {op} {number}"
    spec = rng.choice(SPECS)
    return expression, spec


def eval_expression(expression: str) -> int:
    char_part, op, number_part = expression.split()
    base = ord(char_part.strip("'"))
    number = int(number_part)
    if op == "+":
        return base + number
    return base - number


def check_answer(expression: str, spec: str) -> str:
    value = eval_expression(expression)
    if spec == "%c":
        return chr(value % 256)
    if spec == "%d":
        return str(value)
    if spec == "%x":
        return format(value, "x")
    if spec == "%o":
        return format(value, "o")
    return format(value, "b")


class Module_1_Submodule_2_task_2(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expression, self.spec = generate_params(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            "Чему равно значение выражения "
            f"{self.expression}, если его вывести с помощью спецификатора {self.spec}?"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.expression, self.spec)
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
            expected = check_answer(self.expression, self.spec)
            student = self.student_solution.strip()
            if self._compare_default(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
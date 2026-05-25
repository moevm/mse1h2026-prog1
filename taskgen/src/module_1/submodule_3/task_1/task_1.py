from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


OPS = ["+", "-", "*"]


def generate_arith_expr(rng: random.Random) -> str:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    c = rng.randint(1, 9)
    op1 = rng.choice(OPS)
    op2 = rng.choice(OPS)
    return f"{a} {op1} {b} {op2} {c}"


def safe_eval(expr: str) -> int:
    allowed = set("0123456789+-* ")
    if not set(expr).issubset(allowed):
        raise ValueError("Unsupported expression")
    return int(eval(expr, {"__builtins__": {}}, {}))


def generate_params(seed: int) -> tuple[str, str, str, str]:
    rng = random.Random(seed)
    value1 = generate_arith_expr(rng)
    value2 = generate_arith_expr(rng)
    value3 = generate_arith_expr(rng)
    expr_templates = [
        "A + B * C",
        "A * B - C",
        "A - B + C",
        "A * B + C",
        "A + B - C",
        "A * C - B",
    ]
    expression = rng.choice(expr_templates)
    return value1, value2, value3, expression


def check_answer(value1: str, value2: str, value3: str, expression: str) -> str:
    a = safe_eval(value1)
    b = safe_eval(value2)
    c = safe_eval(value3)
    expr = expression.replace("A", str(a)).replace("B", str(b)).replace("C", str(c))
    return str(safe_eval(expr))


class Module_1_Submodule_3_task_1(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value1, self.value2, self.value3, self.expression = generate_params(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            "Чему будет равно значение выражения "
            f"{self.expression}? Часть программы с определением макросов:\n"
            f"#define A {self.value1}\n"
            f"#define B {self.value2}\n"
            f"#define C {self.value3}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.value1, self.value2, self.value3, self.expression)
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
            expected = check_answer(self.value1, self.value2, self.value3, self.expression)
            student = self.student_solution.strip()
            if self._compare_default(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_expression(seed: int) -> str:
    random.seed(seed)
    num_terms = random.randint(1, 4)
    terms = []
    for _ in range(num_terms):
        a = random.randint(-100, 100)
        b = random.randint(-100, 100)
        while b == 0:
            b = random.randint(-100, 100)
        terms.append(f"({a} % {b})")
    return " + ".join(terms)


def c_mod(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return a - (a // b) * b


def evaluate_expression(expr: str) -> int:
    terms = expr.split('+')
    total = 0
    for term in terms:
        term = term.strip()
        inner = term[1:-1].strip()
        a_str, b_str = inner.split('%')
        a = int(a_str.strip())
        b = int(b_str.strip())
        total += c_mod(a, b)
    return total


def compare_results(student: str, expected: int) -> bool:
    try:
        stud_val = int(student.strip())
        return stud_val == expected
    except ValueError:
        return False


class Module_1_Submodule_5_task_4(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expression = generate_expression(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return f"Чему будет равно значение выражения `{self.expression}`?"

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = evaluate_expression(self.expression)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=str(expected),
                compare_func=lambda out, exp: compare_results(out, int(exp))
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
            expected = evaluate_expression(self.expression)
            student = self.student_solution.strip()
            if compare_results(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
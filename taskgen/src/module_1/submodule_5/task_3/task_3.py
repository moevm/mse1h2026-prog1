from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_expression(seed: int) -> str:
    random.seed(seed)
    num_terms = random.randint(1, 4)
    terms = []
    for _ in range(num_terms):
        is_float1 = random.choice([True, False])
        is_float2 = random.choice([True, False])
        num1 = round(random.uniform(-20, 20), 2) if is_float1 else random.randint(-20, 20)
        num2 = round(random.uniform(-20, 20), 2) if is_float2 else random.randint(-20, 20)
        while abs(num2) < 1e-9:
            num2 = round(random.uniform(-20, 20), 2) if is_float2 else random.randint(-20, 20)
        if isinstance(num1, float) and num1.is_integer():
            num1 = int(num1)
        if isinstance(num2, float) and num2.is_integer():
            num2 = int(num2)
        term = f"({num1} / {num2})"
        terms.append(term)
    return " + ".join(terms)


def evaluate_expression(expr: str) -> str:
    try:
        result = eval(expr)
    except ZeroDivisionError:
        result = float('inf')
    if abs(result - round(result)) < 1e-9:
        return f"{int(result)}.0"
    else:
        s = f"{result:.10f}".rstrip('0').rstrip('.')
        return s if s else "0.0"


def compare_results(student: str, expected: str) -> bool:
    try:
        stud = float(student.strip())
        exp = float(expected)
        return abs(stud - exp) < 1e-6
    except ValueError:
        return False


class Module_1_Submodule_5_task_3(BaseTaskClass):
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
                expected=expected,
                compare_func=compare_results
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
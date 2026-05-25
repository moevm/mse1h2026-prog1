from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_expression(seed: int, num_terms: int = None) -> str:
    random.seed(seed)
    if num_terms is None:
        num_terms = random.randint(1, 3)
    terms = []
    for _ in range(num_terms):
        mantissa = round(random.uniform(0.1, 9.9), random.randint(1, 3))
        exponent = random.randint(-5, 5)
        terms.append(f"{mantissa}e{exponent}")
    return " + ".join(terms)


def evaluate_expression(expr: str) -> float:
    terms = expr.split("+")
    total = 0.0
    for term in terms:
        term = term.strip()
        if 'e' in term or 'E' in term:
            total += float(term)
        else:
            total += float(term)
    return total


def format_result(value: float) -> str:
    s = f"{value:.10f}".rstrip('0').rstrip('.')
    return s


def compare_decimals(student: str, expected: str) -> bool:
    try:
        student_val = float(student.strip())
        expected_val = float(expected)
        return abs(student_val - expected_val) < 1e-9
    except ValueError:
        return False


class Module_1_Submodule_3_task_3(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expression = generate_expression(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (f"Напишите, как выглядит число `{self.expression}` в привычной десятичной форме. "
                f"Число записано в экспоненциальной форме и содержит дробную часть. "
                f"Ввод пуст. Пример: `2.5e-2 = 0.025`. Для суммы нескольких чисел вычислите результат.")

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        total = evaluate_expression(self.expression)
        expected = format_result(total)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=compare_decimals
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
            total = evaluate_expression(self.expression)
            expected = format_result(total)
            student = self.student_solution.strip()
            if compare_decimals(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
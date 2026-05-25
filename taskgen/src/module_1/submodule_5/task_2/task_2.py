from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_expression(seed: int) -> str:
    random.seed(seed)
    operators = ['+', '-', '*', '/', '%']
    num_operators = random.randint(1, 2)
    operands = []
    ops = []
    for i in range(num_operators + 1):
        is_int = random.choice([True, False])
        if is_int:
            value = random.randint(1, 100)
            cast = random.choice([None, '(int)', '(float)'])
            if cast:
                operands.append(f"{cast}{value}")
            else:
                operands.append(str(value))
        else:
            value = round(random.uniform(0.5, 50.0), 2)
            cast = random.choice([None, '(int)', '(float)'])
            if cast:
                operands.append(f"{cast}{value}")
            else:
                operands.append(str(value))
    for j in range(num_operators):
        while True:
            op = random.choice(operators)
            if op == '%':
                left_ok = '.' not in operands[j] and '(float)' not in operands[j]
                right_ok = '.' not in operands[j+1] and '(float)' not in operands[j+1]
                if left_ok and right_ok:
                    ops.append(op)
                    break
            else:
                ops.append(op)
                break
    expr = operands[0]
    for op, operand in zip(ops, operands[1:]):
        expr += f" {op} {operand}"
    return expr


def convert_to_python(expr: str) -> str:
    expr = re.sub(r'\(int\)\s*([\d\.]+)', r'int(\1)', expr)
    expr = re.sub(r'\(float\)\s*([\d\.]+)', r'float(\1)', expr)
    return expr


def evaluate_expression(expr: str) -> str:
    py_expr = convert_to_python(expr)
    try:
        result = eval(py_expr)
    except ZeroDivisionError:
        result = float('inf')
    if abs(result - round(result)) < 1e-9:
        return f"{int(result)}.0"
    else:
        return f"{result:.10f}".rstrip('0').rstrip('.') or "0.0"


def compare_results(student: str, expected: str) -> bool:
    try:
        stud = float(student.strip())
        exp = float(expected)
        return abs(stud - exp) < 1e-6
    except ValueError:
        return False


class Module_1_Submodule_5_task_2(BaseTaskClass):
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
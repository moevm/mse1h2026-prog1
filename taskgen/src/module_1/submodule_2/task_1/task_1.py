from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_true_expr(used: set, rng: random.Random) -> str:
    while True:
        choice = rng.choice(['num', 'sum', 'diff'])
        if choice == 'num':
            val = rng.randint(1, 20)
            expr = str(val)
        elif choice == 'sum':
            a = rng.randint(0, 20)
            b = rng.randint(0, 20 - a)
            if a + b == 0:
                continue
            expr = f"{a} + {b}"
        else:
            a = rng.randint(1, 20)
            b = rng.randint(0, a - 1)
            expr = f"{a} - {b}"
        if expr not in used:
            used.add(expr)
            return expr


def generate_false_expr(used: set, rng: random.Random, null_used: bool) -> tuple[str, bool]:
    types = ['zero', 'zero_zero', 'self_diff']
    if not null_used:
        types.append('null')
    rng.shuffle(types)
    for typ in types:
        if typ == 'zero':
            expr = "0"
            if expr not in used:
                used.add(expr)
                return expr, null_used
        elif typ == 'null':
            expr = "NULL"
            if expr not in used:
                used.add(expr)
                return expr, True
        elif typ == 'zero_zero':
            expr = "0 + 0"
            if expr not in used:
                used.add(expr)
                return expr, null_used
        elif typ == 'self_diff':
            a = rng.randint(1, 10)
            expr = f"{a} - {a}"
            if expr not in used:
                used.add(expr)
                return expr, null_used
    while True:
        a = rng.randint(1, 10)
        expr = f"{a} - {a}"
        if expr not in used:
            used.add(expr)
            return expr, null_used


def generate_expressions(seed: int) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    true_count = rng.randint(1, 3)
    used = set()
    true_exprs = []
    for _ in range(true_count):
        expr = generate_true_expr(used, rng)
        true_exprs.append(expr)

    false_exprs = []
    null_used = False
    while len(false_exprs) < (8 - true_count):
        expr, null_used = generate_false_expr(used, rng, null_used)
        false_exprs.append(expr)

    all_exprs = true_exprs + false_exprs
    rng.shuffle(all_exprs)
    true_indices = [i + 1 for i, expr in enumerate(all_exprs) if expr in true_exprs]
    return all_exprs, true_indices


def check_answer(true_indices: list[int]) -> str:
    return " ".join(map(str, true_indices))


class Module_1_Submodule_2_task_1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expressions, self.true_indices = generate_expressions(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        lines = [
            "Какие из ниже представленных выражений являются истинными?",
            "Ответы укажите через пробел.",
        ]
        lines.extend(f"{i}. {expr}" for i, expr in enumerate(self.expressions, start=1))
        return "\n".join(lines)

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.true_indices)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(" ".join(output.strip().split()), exp)
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
            expected = check_answer(self.true_indices)
            student = self.student_solution.strip()
            student_norm = " ".join(student.split())
            if student_norm == expected:
                return True, "OK: Верный ответ."
            else:
                return False, f"FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
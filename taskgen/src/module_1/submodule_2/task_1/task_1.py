from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def make_number(rng: random.Random) -> tuple[str, bool]:
    value = rng.randint(-20, 20)
    return str(value), value != 0


def make_sum(rng: random.Random) -> tuple[str, bool]:
    left = rng.randint(-20, 20)
    right = rng.randint(-20, 20)
    return f"{left} + {right}", (left + right) != 0


def make_diff(rng: random.Random) -> tuple[str, bool]:
    left = rng.randint(-20, 20)
    right = rng.randint(-20, 20)
    return f"{left} - {right}", (left - right) != 0


def make_null() -> tuple[str, bool]:
    return "NULL", False


def generate_expressions(seed: int) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    makers = [make_number, make_sum, make_diff]

    expressions: list[tuple[str, bool]] = [make_null()]
    while len(expressions) < 8:
        maker = rng.choice(makers + [make_null])
        expr = maker(rng) if maker is not make_null else make_null()
        expressions.append(expr)

    rng.shuffle(expressions)

    true_indices = [i + 1 for i, (_, is_true) in enumerate(expressions) if is_true]
    if not true_indices:
        expressions[0] = ("1", True)
        true_indices = [1]

    return [expr for expr, _ in expressions], true_indices


def check_answer(true_indices: list[int]) -> str:
    return " ".join(map(str, true_indices))


class Module_1_Submodule_2_task_1(BaseTaskClass):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.expressions, self.true_indices = generate_expressions(self.seed)

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
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

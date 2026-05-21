from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    fragment: str
    expected: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        fragment="atexit(A); atexit(B); atexit(C);",
        expected="C B A",
    ),
    1: VariantSpec(
        fragment="atexit(log); atexit(cleanup); atexit(notify);",
        expected="notify cleanup log",
    ),
    2: VariantSpec(
        fragment="atexit(free_mem); atexit(save_log);",
        expected="save_log free_mem",
    ),
    3: VariantSpec(
        fragment="atexit(X); atexit(Y); atexit(Z); atexit(W);",
        expected="W Z Y X",
    ),
}


class AtexitTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# stdlib: atexit\n\n"
            "### Задание №10\n\n"
            "- **Формулировка:**  \n"
            "  Дан фрагмент программы (соответствует вашему варианту).  \n"
            "  Введите имена функций через пробел в том порядке, в котором они\n"
            "  будут вызваны при завершении программы.\n\n"
            f"**Фрагмент:**\n```c\n{v.fragment}\n```\n\n"
            "**Ваш ответ (через пробел):**\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="[скрыто]",
                expected=self.variant.expected,
                compare_func=self._compare_default,
            )
        ]

    def _check_solution_text(self) -> bool:
        normalized = re.sub(r'\s+', ' ', self.solution.strip())
        return normalized == self.variant.expected

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
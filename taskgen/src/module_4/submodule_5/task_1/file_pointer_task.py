from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    initializer: str
    description: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(initializer="NULL", description="нулевой (не открытый) файловый указатель"),
    1: VariantSpec(initializer="stdin", description="стандартный ввод"),
    2: VariantSpec(initializer="stdout", description="стандартный вывод"),
    3: VariantSpec(initializer="stderr", description="стандартный вывод ошибок"),
}


class FilePointerTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Объявление переменной типа FILE *\n\n"
            "### Задание №1\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **одну строку** — объявление переменной типа `FILE *` с именем `fp`\n"
            "  и её инициализацией значением, соответствующим вашему варианту.\n\n"
            "  **Ваш вариант:** " + v.description + "\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="[скрыто]",
                expected="ok",
                compare_func=self._compare_default,
            )
        ]

    def _check_solution_text(self) -> bool:
        v = self.variant
        text = self.solution.strip().rstrip(";").strip()
        pattern = r'^\s*FILE\s*\*\s*fp\s*=\s*(' + re.escape(v.initializer) + r')\s*$'
        return bool(re.search(pattern, text))

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
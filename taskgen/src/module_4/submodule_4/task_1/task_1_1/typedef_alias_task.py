from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    base_type: str
    alias: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(base_type="unsigned char",       alias="u8"),
    1: VariantSpec(base_type="unsigned short",      alias="u16"),
    2: VariantSpec(base_type="unsigned long long",  alias="u64"),
    3: VariantSpec(base_type="long long",           alias="i64"),
}


class TypedefAliasTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Typedef: псевдоним для целочисленного типа\n\n"
            "### Задание №1.1\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **одну строку** — объявление псевдонима для целочисленного типа:  \n\n"
            f"  Базовый тип: `{v.base_type}`  \n"
            f"  Псевдоним: `{v.alias}`  \n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"[скрыто]",
                expected="ok",
                compare_func=self._compare_default,
            )
        ]

    def _check_solution_text(self) -> bool:
        v = self.variant
        text = self.solution.strip().rstrip(";").strip()
        expected = f"typedef {v.base_type} {v.alias}"
        text_norm = re.sub(r'\s+', ' ', text)
        expected_norm = re.sub(r'\s+', ' ', expected)
        return text_norm == expected_norm

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), f"[скрыто]"
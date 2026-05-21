from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    filename: str
    mode: str
    description: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(filename="data.txt", mode="r", description="открыть текстовый файл для чтения"),
    1: VariantSpec(filename="out.txt", mode="w", description="открыть текстовый файл для записи"),
    2: VariantSpec(filename="log.txt", mode="a", description="открыть текстовый файл для дозаписи в конец"),
    3: VariantSpec(filename="data.bin", mode="rb", description="открыть бинарный файл для чтения"),
    4: VariantSpec(filename="out.bin", mode="wb", description="открыть бинарный файл для записи"),
    5: VariantSpec(filename="data.txt", mode="r+", description="открыть текстовый файл для чтения и записи"),
}


class FOpenTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Вызов функции открытия файла\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **одну строку** — вызов функции открытия файла с присваиванием результата\n"
            "  переменной `FILE *fp`, соответствующий вашему варианту.\n\n"
            f"  **Имя файла:** `{v.filename}`  \n"
            f"  **Пояснение:** {v.description} (по нему выберите режим)\n\n"
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
        pattern = (
            r'^\s*FILE\s*\*\s*fp\s*=\s*fopen\s*\(\s*"'
            + re.escape(v.filename)
            + r'"\s*,\s*"'
            + re.escape(v.mode)
            + r'"\s*\)\s*$'
        )
        return bool(re.search(pattern, text))

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
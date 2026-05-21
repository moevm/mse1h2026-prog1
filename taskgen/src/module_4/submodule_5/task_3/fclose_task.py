from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    var_name: str
    description: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(var_name="fp",   description='fopen("data.txt", "r")'),
    1: VariantSpec(var_name="fin",  description='fopen("input.bin", "rb")'),
    2: VariantSpec(var_name="fout", description='fopen("output.txt", "w")'),
    3: VariantSpec(var_name="flog", description='fopen("log.txt", "a")'),
}


class FCloseTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Вызов функции закрытия файла\n\n"
            "### Задание №3\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **одну строку** — вызов функции закрытия файла для переменной,\n"
            "  соответствующей вашему варианту.\n\n"
            f"  **Имя переменной:** `{v.var_name}`  \n"
            f"  **Переменная открыта как:** `{v.description}`\n\n"
            "  Подключать заголовочные файлы и писать `main()` не нужно.\n"
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
        pattern = r'^\s*fclose\s*\(\s*' + re.escape(v.var_name) + r'\s*\)\s*$'
        return bool(re.search(pattern, text))

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
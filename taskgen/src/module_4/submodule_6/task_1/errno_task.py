from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    filename: str
    condition: str
    expected: str      # числовой ответ в виде строки


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        filename="data.txt",
        condition="Файл `data.txt` не существует",
        expected="2",        # ENOENT
    ),
    1: VariantSpec(
        filename="/root/secret.txt",
        condition="Файл `/root/secret.txt` существует, но у процесса нет прав на чтение",
        expected="13",       # EACCES
    ),
}


class ErrnoTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        code = f"""#include <stdio.h>
#include <errno.h>

int main(void) {{
    errno = 0;
    FILE *f = fopen("{v.filename}", "r");
    int saved_err = errno;
    if (f) fclose(f);
    return 0;
}}"""
        return (
            "# Обработка ошибок: errno\n\n"
            "### Задание №1\n\n"
            "- **Формулировка:**  \n"
            "  Вам дан фрагмент кода на C и описание условий его выполнения (соответствует вашему варианту).\n"
            "  Определите, какое числовое значение будет у переменной `saved_err` после выполнения этого кода\n"
            "  на платформе Linux (x86-64).\n"
            "  Введите одно целое число.\n\n"
            f"**Условие выполнения:** {v.condition}\n\n"
            "**Фрагмент кода:**\n"
            f"```c\n{code}\n```\n\n"
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
        # Убираем пробелы и пустые строки
        answer = self.solution.strip()
        return answer == self.variant.expected

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
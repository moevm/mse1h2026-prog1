from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    description: str
    command: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(description="Вывести текущую дату и время", command="date"),
    1: VariantSpec(description="Вывести список файлов текущей директории", command="ls"),
    2: VariantSpec(description="Вывести имя текущего пользователя", command="whoami"),
    3: VariantSpec(description="Вывести путь к текущей рабочей директории", command="pwd"),
}


class SystemTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# stdlib: system\n\n"
            "### Задание №12\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **одну строку** на C — вызов стандартной библиотечной функции,\n"
            "  которая передаёт команду командному процессору операционной системы\n"
            "  и возвращает код её завершения.\n"
            "  Результат сохраните в переменную `ret` (уже объявлена).\n\n"
            f"**Ваш вариант:** {v.description}\n\n"
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
        # разрешаем пробелы: ret = system("команда")
        pattern = r'^\s*ret\s*=\s*system\s*\(\s*"' + re.escape(v.command) + r'"\s*\)\s*$'
        return bool(re.search(pattern, text))

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
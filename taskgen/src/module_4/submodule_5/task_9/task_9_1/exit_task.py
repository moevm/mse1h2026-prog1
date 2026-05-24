from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


class ExitTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_task(self) -> str:
        return (
            "# stdlib: exit — успешное завершение\n\n"
            "### Задание №9.1\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **одну строку** — вызов `exit` для штатного завершения программы.\n\n"
            "  Используйте именованный макрос из `<stdlib.h>`, а не числовую константу.\n"
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
        text = self.solution.strip().rstrip(";").strip()
        pattern = r'^\s*exit\s*\(\s*EXIT_SUCCESS\s*\)\s*$'
        return bool(re.search(pattern, text))

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
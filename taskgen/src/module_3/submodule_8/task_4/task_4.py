from typing import Optional
import re
import os
import subprocess
import tempfile
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task4(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 2
        if rem == 0:
            return {
                "func": "float_to_int_bits", "src": "float", "dst": "int",
                "fmt": "As int: %d\\n", "test_val": "3.14f", "expected_out": "As int: 1078523331\n"
            }
        else:
            return {
                "func": "int_to_float_bits", "src": "int", "dst": "float",
                "fmt": "As float: %g\\n", "test_val": "1078523331", "expected_out": "As float: 3.14\n"
            }

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>

void {p["func"]}({p["src"]} val);

int main() {{
    {p["func"]}({p["test_val"]});
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Нарушение strict aliasing
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["src"]} val)`, которая интерпретирует битовое представление числа `val` как тип `{p["dst"]}` и выводит его.
Запрещено использовать приведение типов через указатели: `*({p["dst"]}*)&val`.
Используйте безопасный способ: `memcpy` или `union`.

**Формат вывода:** `{p["fmt"].strip()}`
"""

    def _generate_tests(self):
        p = self._get_params()
        self.tests = [TestItem(
            input_str="",
            showed_input="[скрыто]",
            expected=p["expected_out"],
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["src"]}\s+val\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}({p['src']} val)`."

        if re.search(rf'\*\s*\(\s*\({p["dst"]}\s*\*\)\s*&\s*val\s*\)', code):
            return f"Ошибка: запрещено приведение через указатель `*({p['dst']}*)&val`."

        has_memcpy = re.search(r'memcpy\s*\(', code)
        has_union = re.search(r'\bunion\b', code) and re.search(r'\.\s*\w+\s*=', code)
        
        if not (has_memcpy or has_union):
            return "Ошибка: используйте `memcpy` или `union` для безопасной реинтерпретации битов."

        fmt_escaped = re.escape(p["fmt"].replace("\n", r"\n"))
        if not re.search(rf'printf\s*\(.*{fmt_escaped}', code, re.DOTALL):
            return f"Ошибка: вывод должен соответствовать формату `{p['fmt'].strip()}`."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
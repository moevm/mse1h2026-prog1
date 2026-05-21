from typing import Optional
import re
import os
import subprocess
import tempfile
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule6_Task3(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "calc_via_table", "type": "int", "fmt": "Result: %d\n"}
        elif rem == 1:
            return {"func": "execute_op", "type": "float", "fmt": "Output: %.1f\n"}
        else:
            return {"func": "dispatch", "type": "int", "fmt": "=> %d\n"}

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>

void {p["func"]}(int op, {p["type"]} a, {p["type"]} b);

int main() {{
    {p["func"]}(0, 10, 5);   // 15
    {p["func"]}(1, 20, 8);   // 12
    {p["func"]}(2, 3, 7);    // 21
    {p["func"]}(3, 100, 4);  // 25
    {p["func"]}(3, 10, 0);   // 0 (деление на ноль)
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        fmt = f"{p['fmt'].replace('\n', '\\n')}"
        return f"""### Тема: Таблицы функций (массивы указателей)
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}(int op, {p["type"]} a, {p["type"]} b)`, которая выбирает и вызывает функцию из таблицы по индексу `op` и выводит результат в формате `{fmt.strip()}`.

**Таблица операций (индекс - функция):**
| `op` | Операция | Реализация |
|------|----------|-----------|
| 0 | Сложение | `a + b` |
| 1 | Вычитание | `a - b` |
| 2 | Умножение | `a * b` |
| 3 | Деление | `b != 0 ? a / b : 0` |

**Запрещено** использовать `switch`/`if-else` для выбора операции. Используйте массив указателей на функции.

**Формат вывода:** `{fmt.strip()}`
"""

    def _generate_tests(self):
        p = self._get_params()
        cases = [(0, 10, 5), (1, 20, 8), (2, 3, 7), (3, 100, 4), (3, 10, 0)]
        lines = []
        for op, a, b in cases:
            if op == 0: val = a + b
            elif op == 1: val = a - b
            elif op == 2: val = a * b
            elif op == 3: val = (a / b) if b != 0 else 0
            else: val = 0

            if p["type"] == "float":
                lines.append(p["fmt"] % float(val))
            else:
                lines.append(p["fmt"] % val)
        
        expected = "".join(lines)
        self.tests = [TestItem(
            input_str="",
            showed_input="[скрыто]",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*int\s+op\s*,\s*{p["type"]}\s+a\s*,\s*{p["type"]}\s+b\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}(int op, {p['type']} a, {p['type']} b)`."

        if re.search(r'\bswitch\b', code):
            return "Ошибка: использование switch запрещено. Используйте массив указателей на функции."

        if not re.search(rf'\(\s*\*\s*\w+\s*\[\s*\d*\s*\]\s*\)\s*\(\s*{p["type"]}\s*,\s*{p["type"]}\s*\)', code):
            return "Ошибка: необходимо объявить массив указателей на функции, например `type (*table[])(type, type)`."

        if not re.search(rf'\w+\s*\[\s*op\s*\]\s*\(', code):
            return "Ошибка: вызов должен выполняться через индекс массива: `table[op](a, b)`."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected) 
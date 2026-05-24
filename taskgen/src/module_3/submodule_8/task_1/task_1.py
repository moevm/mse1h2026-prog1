from typing import Optional
import re
import os
import subprocess
import tempfile
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task1(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "safe_deref_int", "type": "int", 
                    "ok": "Value: ", "null": "NULL ptr: ", "fmt": "%d\n", 
                    "def": "0", "test_val": "42"}
        elif rem == 1:
            return {"func": "read_float_safe", "type": "float", 
                    "ok": "Got: ", "null": "Error: ", "fmt": "%.1f\n", 
                    "def": "0.0f", "test_val": "3.14"}
        else:
            return {"func": "get_char_checked", "type": "char", 
                    "ok": "Char: ", "null": "No data: ", "fmt": "%c\n", 
                    "def": "'\\0'", "test_val": "'Z'"}

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>
#include <stdlib.h>

{p["type"]} {p["func"]}({p["type"]} *ptr);

int main() {{
    // Тест 1: NULL-указатель
    {p["func"]}(NULL);

    // Тест 2: Валидный указатель
    {p["type"]} val = {p["test_val"]};
    {p["func"]}(&val);

    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        fmt_null = f"{p['null']}{p['fmt'].replace('\n', '\\n')}"
        fmt_ok = f"{p['ok']}{p['fmt'].replace('\n', '\\n')}"
        
        return f"""### Тема: Разыменование `NULL`
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} *ptr)`, которая безопасно разыменовывает указатель. 
- Если `ptr == NULL`, функция должна вывести `{p["null"]}` и вернуть `{p["def"]}`.
- В противном случае — вывести `{p["ok"]}` и вернуть значение `*ptr`.

**Формат вывода:**
- При `ptr == NULL`: `{fmt_null}` 
- При `ptr != NULL`: `{fmt_ok}` 
"""

    def _generate_tests(self):
        p = self._get_params()
        if p["type"] == "int":
            expected = f"{p['null']}0\n{p['ok']}42\n"
        elif p["type"] == "float":
            expected = f"{p['null']}0.0\n{p['ok']}3.1\n"
        else:
            expected = f"{p['null']}\n{p['ok']}Z\n"

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

        sig = rf'{p["type"]}\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\s*ptr\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}({p['type']} *ptr)`."

        if not re.search(r'if\s*\(\s*ptr\s*==\s*NULL\s*\)|if\s*\(\s*!ptr\s*\)', code):
            return "Ошибка: необходимо проверить указатель на NULL перед разыменованием."

        def_val = p["def"].strip("'")
        if not re.search(rf'return\s+(?:{re.escape(p["def"])}|0)\s*;', code):
            return f"Ошибка: при NULL необходимо вернуть значение по умолчанию ({p['def']})."

        if not re.search(r'return\s+\*ptr\s*;', code):
            return "Ошибка: необходимо вернуть разыменованное значение `*ptr`."

        if p["null"] not in code or p["ok"] not in code:
            return "Ошибка: вывод должен содержать сообщения для NULL и валидного указателя."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
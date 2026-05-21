from typing import Optional
import re
import os
import subprocess
import tempfile
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task11(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "safe_free_int", "type": "int", "ptr": "buffer"}
        elif rem == 1:
            return {"func": "safe_free_float", "type": "float", "ptr": "data"}
        else:
            return {"func": "safe_free_char", "type": "char", "ptr": "str"}

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>
#include <stdlib.h>

void {p["func"]}({p["type"]} **{p["ptr"]});

int main() {{
    {p["type"]} *{p["ptr"]} = malloc(sizeof({p["type"]}));
    if (!{p["ptr"]}) return 1;

    printf("Before 1st: %s\\n", {p["ptr"]} == NULL ? "NULL" : "ALIVE");
    {p["func"]}(&{p["ptr"]});
    printf("After 1st: %s\\n", {p["ptr"]} == NULL ? "NULL" : "ALIVE");

    {p["func"]}(&{p["ptr"]});
    printf("After 2nd: %s\\n", {p["ptr"]} == NULL ? "NULL" : "ALIVE");

    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Защита от Double Free
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} **{p["ptr"]})`, которая безопасно освобождает память и предотвращает двойное освобождение.
Функция должна:
1. Проверить, что `{p["ptr"]}` не равен `NULL` и не указывает на `NULL`.
2. Освободить память через `free()`.
3. Присвоить `*{p["ptr"]}` значение `NULL`, чтобы указатель в вызывающем коде стал безопасным.
"""

    def _generate_tests(self):
        expected = "Before 1st: ALIVE\nAfter 1st: NULL\nAfter 2nd: NULL\n"
        self.tests = [TestItem(
            input_str="",
            showed_input="",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\*\s*{p["ptr"]}\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: сигнатура должна быть `void {p['func']}({p['type']} **{p['ptr']})`."

        null_guard = rf'if\s*\(\s*{p["ptr"]}\s*&&\s*\*{p["ptr"]}\s*\)'
        if not re.search(null_guard, code):
            return f"Ошибка: обязательно проверьте `{p['ptr']}` и `*{p['ptr']}` на NULL перед освобождением."

        if not re.search(rf'free\s*\(\s*\*\s*{p["ptr"]}\s*\)', code):
            return f"Ошибка: вызовите `free(*{p['ptr']})`."

        if not re.search(rf'\*\s*{p["ptr"]}\s*=\s*NULL', code):
            return f"Ошибка: после `free` обязательно присвойте `*{p['ptr']} = NULL`."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        norm = lambda s: s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return norm(output) == norm(expected)
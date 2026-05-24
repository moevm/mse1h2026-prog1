from typing import Optional
import re
import os
import subprocess
import tempfile
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task3(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "create_zero_array", "type": "int", "init": "0"}
        elif rem == 1:
            return {"func": "init_float_array", "type": "float", "init": "0.0f"}
        else:
            return {"func": "alloc_cleared_buf", "type": "char", "init": "0"}

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>
#include <stdlib.h>

{p["type"]} *{p["func"]}(int size);

int main() {{
    int size = 10;
    {p["type"]} *arr = {p["func"]}(size);

    if (!arr) {{
        printf("Status: FAIL (NULL)\\n");
        return 1;
    }}

    int ok = 1;
    for (int i = 0; i < size; i++) {{
        if (arr[i] != 0) {{
            ok = 0;
            break;
        }}
    }}

    if (ok) printf("Status: OK\\n");
    else printf("Status: FAIL\\n");

    free(arr);
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Использование неинициализированной памяти
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{p["func"]}(int size)`, которая выделяет память для массива из `size` элементов типа `{p["type"]}` и гарантирует, что все элементы равны `{p["init"]}` до начала использования. Верните указатель на выделенную память. 
"""

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str="",
            showed_input="",
            expected="",
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'{p["type"]}\s*\*\s*{p["func"]}\s*\(\s*int\s+size\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}(int size)`."

        if not re.search(r'calloc\s*\(', code):
            return "Ошибка: используйте `calloc` для гарантированной инициализации нулём. `malloc` оставляет память неинициализированной."

        if not re.search(r'return\s+\w+\s*;', code):
            return "Ошибка: функция должна возвращать указатель на выделенную память."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
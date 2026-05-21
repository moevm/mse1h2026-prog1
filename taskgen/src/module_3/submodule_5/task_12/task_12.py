from typing import Optional
import re
import os
import subprocess
import tempfile
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task12(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "safe_int_ptr", "type": "int", "val": "val", "fmt": "%d", "def": "42"}
        elif rem == 1:
            return {"func": "safe_float_ptr", "type": "float", "val": "x", "fmt": "%.2f", "def": "3.14f"}
        else:
            return {"func": "safe_char_ptr", "type": "char", "val": "c", "fmt": "%c", "def": "'A'"}

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>
#include <stdlib.h>

{p["type"]} *{p["func"]}({p["type"]} {p["val"]});

int main() {{
    {p["type"]} test_val = {p["def"]};
    {p["type"]} *ptr = {p["func"]}(test_val);
    if (ptr) {{
        printf("{p["fmt"]}", *ptr);
        free(ptr);
    }} else {{
        printf("NULL\\n");
    }}
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Висячий указатель
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} {p["val"]})`, которая возвращает указатель на переданное значение `{p["val"]}`.
Запрещено возвращать адрес локальной переменной (`return &{p["val"]};`) — это создаст висячий указатель.
Обязательно используйте динамическое выделение памяти (`malloc`/`calloc`), скопируйте значение и верните указатель.
При ошибке выделения верните `NULL`.
"""

    def _generate_tests(self):
        p = self._get_params()
        if p["type"] == "int":
            expected = p["fmt"] % 42
        elif p["type"] == "float":
            expected = p["fmt"] % 3.14
        else:
            expected = p["fmt"] % 'A'

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

        sig = rf'{p["type"]}\s*\*\s+{p["func"]}\s*\(\s*{p["type"]}\s+{p["val"]}\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['type']} *{p['func']}({p['type']} {p['val']})`."

        if re.search(rf'return\s+&\s*{p["val"]}\s*;', code):
            return f"Ошибка: запрещено возвращать адрес параметра `&{p['val']}`."

        if not re.search(r'malloc\s*\(|calloc\s*\(', code):
            return "Ошибка: необходимо использовать динамическое выделение памяти (malloc/calloc)."

        alloc_match = re.search(r'(\w+)\s*=\s*(?:malloc|calloc)\s*\(', code)
        if alloc_match:
            ptr_var = alloc_match.group(1)
            null_checks = [
                rf'if\s*\(\s*!{ptr_var}\s*\)',           
                rf'if\s*\(\s*{ptr_var}\s*\)',            
                rf'if\s*\(\s*{ptr_var}\s*==\s*NULL\s*\)',
                rf'if\s*\(\s*NULL\s*==\s*{ptr_var}\s*\)',
                rf'if\s*\(\s*{ptr_var}\s*!=\s*NULL\s*\)',
                rf'if\s*\(\s*NULL\s*!=\s*{ptr_var}\s*\)',
                rf'if\s*\(\s*!{ptr_var}\s*\)',          
            ]
            if not any(re.search(pat, code) for pat in null_checks):
                return f"Ошибка: необходимо проверить переменную `{ptr_var}` (результат malloc) на NULL."

        if not re.search(rf'\*\s*\w+\s*=\s*{p["val"]}\b', code):
            return f"Ошибка: значение `{p['val']}` должно быть скопировано в выделенную память."

        if not re.search(r'return\s+\w+\s*;', code):
            return "Ошибка: функция должна возвращать указатель."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
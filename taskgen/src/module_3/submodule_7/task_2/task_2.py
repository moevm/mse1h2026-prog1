from typing import Optional
import re
import os
import sys
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule7_Task2(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {
                "func": "copy_ints", "src_t": "int", "dst_t": "int",
                "src_p": "src", "dst_p": "dst", "fmt": "Copy[%d]: %d -> %d\n"
            }
        elif rem == 1:
            return {
                "func": "scale_floats", "src_t": "float", "dst_t": "float",
                "src_p": "input", "dst_p": "output", "fmt": "Scaled[%d]: %.1f -> %.1f\n"
            }
        else:
            return {
                "func": "mirror_chars", "src_t": "char", "dst_t": "char",
                "src_p": "read_buf", "dst_p": "write_buf", "fmt": "Mir[%d]: '%c' -> '%c'\n"
            }

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        if p["src_t"] == "int":
            init_vals = "{10, 20, 30, 40, 50}"
        elif p["src_t"] == "float":
            init_vals = "{1.5, 2.5, 3.5, 4.5, 5.5}"
        else:
            init_vals = "{'A', 'B', 'C', 'D', 'E'}"
            
        size = 5
        return f'''#include <stdio.h>

/* Прототип с const гарантирует проверку контракта на этапе компиляции */
void {p["func"]}(const {p["src_t"]} *{p["src_p"]}, {p["dst_t"]} *{p["dst_p"]}, int size);

int main() {{
    {p["src_t"]} {p["src_p"]}[] = {init_vals};
    {p["dst_t"]} {p["dst_p"]}[{size}];
    {p["func"]}({p["src_p"]}, {p["dst_p"]}, {size});
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        fmt = f"{p['fmt'].replace('\n', '\\n')}"
        return f"""### Тема: const correctness
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}(const {p["src_t"]} *{p["src_p"]}, {p["dst_t"]} *{p["dst_p"]}, int size)`, которая копирует элементы из массива `{p["src_p"]}` в `{p["dst_p"]}` и выводит пары "оригинал → копия" в формате `{fmt.strip()}`.
- Исходный массив `{p["src_p"]}` должен быть доступен **только для чтения** (`const` обязателен).
- Массив `{p["dst_p"]}` должен оставаться **изменяемым** (без `const`).
- Нарушение контракта приведёт к ошибке компиляции в автотесте.

**Формат вывода:** `{fmt.strip()}` (для каждого индекса)
"""

    def _generate_tests(self):
        p = self._get_params()
        if p["src_t"] == "int":
            vals = [10, 20, 30, 40, 50]
        elif p["src_t"] == "float":
            vals = [1.5, 2.5, 3.5, 4.5, 5.5]
        else:
            vals = ['A', 'B', 'C', 'D', 'E']

        lines = []
        for i, v in enumerate(vals):
            lines.append(p["fmt"] % (i, v, v))
        expected = "".join(lines)

        self.tests = [TestItem(
            input_str="",
            showed_input=f"[скрыто]",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig_pat = rf'void\s+{p["func"]}\s*\(\s*(?:const\s+{p["src_t"]}|{p["src_t"]}\s+const)\s*\*\s*{p["src_p"]}\s*,\s*{p["dst_t"]}\s*\*\s*{p["dst_p"]}\s*,\s*int\s+size\s*\)'
        if not re.search(sig_pat, code):
            return f"Ошибка: неверная сигнатура. Требуется `const {p['src_t']} *{p['src_p']}` в первом параметре."

        fmt_escaped = re.escape(p["fmt"].replace("\n", r"\n"))
        if not re.search(rf'printf\s*\(.*{fmt_escaped}', code, re.DOTALL):
            return f"Ошибка: формат вывода должен соответствовать `{p['fmt'].strip()}`."

        return None

    def compile(self) -> Optional[str]:
        custom_args = "-std=c11 -O2 -Wall -Wextra -Werror -Wcast-qual -Wdiscarded-qualifiers"
        return self._compile_internal(
            solution_name="solution.c",
            compiler="gcc",
            compile_args=custom_args,
            keep_static_obj_files=True
        )

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
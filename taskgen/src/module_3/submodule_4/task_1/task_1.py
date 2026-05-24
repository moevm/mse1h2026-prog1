from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task1(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_test_driver_code(self) -> str:
        names = ["my_strlen", "calc_str_len", "get_string_size", "ptr_strlen"]
        func_name = names[self.seed % 4]
        return f'''#include <stdio.h>

int {func_name}(char *s);

int main() {{
    char s[256];
    if (scanf("%255s", s) != 1) return 1;
    printf("%d\\n", {func_name}(s));
    return 0;
}}
'''

    def generate_task(self) -> str:
        names = ["my_strlen", "calc_str_len", "get_string_size", "ptr_strlen"]
        func_name = names[self.seed % 4]
        return f"""### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `int {func_name}(char *s)`, которая возвращает количество символов в строке (без учёта `'\\0'`).
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        base_strings = ["a", "hello", "test123", "C", "programming", "x", "abc", "C_Programming", "data"]

        while len(base_strings) < self.tests_num:
            base_strings.append("".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789_", k=random.randint(1, 20))))

        for s in base_strings[:self.tests_num]:
            input_str = s
            expected = f"{len(s)}\n"
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f's = "{s}"',
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        names = ["my_strlen", "calc_str_len", "get_string_size", "ptr_strlen"]
        func_name = names[self.seed % 4]

        sig_pattern = rf'int\s+{func_name}\s*\(\s*char\s*\*\s*s\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if '#include <string.h>' in self.solution or '#include "string.h"' in self.solution:
            return "Ошибка: запрещено подключать <string.h>."
        if re.search(r'\bstrlen\b', self.solution):
            return "Ошибка: запрещено использовать библиотечную функцию strlen."

        return None

    def _build_program_source(self) -> str:
        return (
            f"{self.solution}\n\n"
            f"{self._get_test_driver_code()}\n"
        )

    def _compile_and_run(self, test_index: int) -> tuple[bool, str]:
        program_source = self._build_program_source()
        test = self.tests[test_index]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", "-Wall", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run(
                [str(exe_path)],
                input=test.input_str.encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            output = "\n".join(
                part for part in (
                    run_proc.stdout.decode().strip(),
                    run_proc.stderr.decode().strip(),
                ) if part
            )
            if run_proc.returncode != 0:
                return False, output
                
            return True, output

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        test_index = self.tests.index(test)
        ok, result = self._compile_and_run(test_index)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
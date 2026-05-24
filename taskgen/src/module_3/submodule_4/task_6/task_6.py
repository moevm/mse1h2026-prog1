from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task6(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_test_driver_code(self) -> str:
        names = ["my_strstr", "custom_strstr", "find_substring", "manual_strstr"]
        func_name = names[self.seed % 4]
        return f'''#include <stdio.h>

char *{func_name}(char *haystack, char *needle);

int main() {{
    char h[512], n[256];
    if (scanf("%511s %255s", h, n) != 2) return 1;

    char *res = {func_name}(h, n);
    if (res != NULL) printf("Found: %s\\n", res);
    else printf("Not found\\n");

    return 0;
}}
'''

    def generate_task(self) -> str:
        names = ["my_strstr", "custom_strstr", "find_substring", "manual_strstr"]
        func_name = names[self.seed % 4]
        return f"""### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *haystack, char *needle)`, которая находит первое вхождение подстроки `needle` в строке `haystack`.
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        base_pairs = [
            ("hello_world", "world"),
            ("abcdef", "xyz"),
            ("programming", "gram"),
            ("data_struct", "a"),
            ("test_case", "test"),
            ("C_language", "lang"),
            ("mississippi", "sip"),
            ("boundary_check", "ary"),
            ("no_match_here", "zzz"),
            ("edge_case", "dge")
        ]

        while len(base_pairs) < self.tests_num:
            h_len = random.randint(5, 15)
            haystack = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=h_len))
            n_len = random.randint(1, min(4, h_len))
            if random.random() < 0.7:
                idx = random.randint(0, h_len - n_len)
                needle = haystack[idx:idx + n_len]
            else:
                needle = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$", k=n_len))
            base_pairs.append((haystack, needle))

        for h, n in base_pairs[:self.tests_num]:
            input_str = f"{h} {n}"
            idx = h.find(n)
            expected = f"Found: {h[idx:]}\n" if idx != -1 else "Not found\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f'haystack="{h}", needle="{n}"',
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        names = ["my_strstr", "custom_strstr", "find_substring", "manual_strstr"]
        func_name = names[self.seed % 4]

        sig_pattern = rf'char\s*\*\s*{func_name}\s*\(\s*char\s*\*\s*haystack\s*,\s*char\s*\*\s*needle\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if '#include <string.h>' in self.solution or re.search(r'\bstrstr\b', self.solution):
            return "Ошибка: запрещено использовать <string.h> или библиотечную strstr."

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
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
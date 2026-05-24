from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task7(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> tuple[str, str, str]:
        configs = [
            ("read_ptr_value", "Value: %d\n", "Value: 0\n"),
            ("print_deref", "Dereferenced: %d\n", "NULL pointer\n"),
            ("show_content", "Content: %d\n", "No data\n"),
            ("fetch_from_ptr", "Ptr value: %d\n", "Empty\n")
        ]
        return configs[self.seed % 4]

    def _get_test_driver_code(self) -> str:
        func, _, _ = self._get_params()
        return f'''#include <stdio.h>

void {func}(int *ptr);

int main() {{
    int val, flag;
    if (scanf("%d %d", &val, &flag) != 2) return 1;
    
    if (flag) {{
        {func}(0);
    }} else {{
        {func}(&val);
    }}
    return 0;
}}
'''

    def generate_task(self) -> str:
        func, print_fmt, null_msg = self._get_params()
        return f"""### Тема: Адрес переменной 

**Сложность:** легкая

**Задача:**
Чтение переменной через указатель
Напишите функцию `void {func}(int *ptr)`, которая принимает указатель на целое число 
и выводит значение, находящееся по этому адресу. 
Если указатель NULL вывести `{null_msg.strip()}`.
Формат вывода:
{print_fmt.strip()}
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        _, print_fmt, null_msg = self._get_params()

        cases = [
            (10, 0), (-7, 0), (0, 0), (42, 0), 
            (0, 1)                               
        ]
        while len(cases) < self.tests_num:
            cases.append((random.randint(-50, 50), 0))

        for val, flag in cases[:self.tests_num]:
            input_str = f"{val} {flag}"
            if flag == 1:
                expected = null_msg
            else:
                expected = print_fmt % val

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"ptr={'NULL' if flag else '&val'}, val={val}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func, _, _ = self._get_params()

        sig_pattern = rf'void\s+{func}\s*\(\s*int\s*\*\s*ptr\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: неверная сигнатура функции `{func}(int *ptr)`."

        if re.search(r'\bNULL\b', self.solution) is None and re.search(r'\b0\b', self.solution) is None:
            if 'if' not in self.solution or 'ptr' not in self.solution:
                return "Ошибка: не найдена проверка указателя на NULL."

        if not re.search(r'\*\s*ptr', self.solution):
            return "Ошибка: не найдено разыменование указателя (*ptr)."

        return None

    def _build_program_source(self) -> str:
        return f"{self.solution}\n\n{self._get_test_driver_code()}"

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
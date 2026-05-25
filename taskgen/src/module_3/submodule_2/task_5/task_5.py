from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task5(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> tuple[str, str, str, str, str]:
        rem3 = self.seed % 3
        func_names = ["sum_via_ptrs", "add_pointed_vals", "compute_sum_ref"]
        p1_names = ["a", "lhs", "val_x"]
        p2_names = ["b", "rhs", "val_y"]
        p3_names = ["res", "out", "target"]
        fmts = ["Calculated: %d\n", "Результат => %d\n", "[Output] %d\n", "Final value -> %d\n"]

        return (
            func_names[rem3],
            p1_names[rem3],
            p2_names[rem3],
            p3_names[rem3],
            fmts[self.seed % 4]
        )

    def _get_test_driver_code(self) -> str:
        func, p1, p2, p3, _ = self._get_params()
        return f'''#include <stdio.h>

void {func}(int *{p1}, int *{p2}, int *{p3});

int main() {{
    int x, y, z;
    if (scanf("%d %d", &x, &y) != 2) return 1;
    {func}(&x, &y, &z);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func, p1, p2, p3, fmt = self._get_params()
        return f"""### Тема: Разыменовывание

**Сложность:** легкая

**Задание:**
Напишите функцию `{func}(int *{p1}, int *{p2}, int *{p3})`, которая разыменовывает указатели `{p1}` и `{p2}`, вычисляет сумму значений и записывает результат по адресу `{p3}`. Внутри функции выведите полученное значение в формате `{fmt}`.
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        _, _, _, _, fmt_str = self._get_params()

        base_pairs = [(10, 7), (-5, 12), (0, 0), (100, -50), (-15, -20)]
        for _ in range(self.tests_num - len(base_pairs)):
            base_pairs.append((random.randint(-20, 50), random.randint(-20, 50)))

        for v1, v2 in base_pairs[:self.tests_num]:
            input_str = f"{v1} {v2}"
            expected = fmt_str.replace("%d", str(v1 + v2))
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"*p1={v1}, *p2={v2}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func, p1, p2, p3, _ = self._get_params()

        sig_pattern = rf'void\s+{func}\s*\(\s*int\s*\*\s*{p1}\s*,\s*int\s*\*\s*{p2}\s*,\s*int\s*\*\s*{p3}\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: неверная сигнатура функции `{func}(int *{p1}, int *{p2}, int *{p3})`."

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
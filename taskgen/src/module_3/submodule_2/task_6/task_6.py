from typing import Optional
import random
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task6(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_test_driver_code(self) -> str:
        if self.seed % 2 == 0:
            return r'''#include <stdio.h>

void change_value(float *ptr, float new_value);

int main() {
    float value, new_val;
    if (scanf("%f %f", &value, &new_val) != 2) {
        return 1;
    }
    change_value(&value, new_val);
    return 0;
}
'''
        else:
            return r'''#include <stdio.h>

void transform_number(int *ptr, int new_value);

int main() {
    int value, new_val;
    if (scanf("%d %d", &value, &new_val) != 2) {
        return 1;
    }
    transform_number(&value, new_val);
    return 0;
}
'''

    def generate_task(self) -> str:
        if self.seed % 2 == 0:
            return """### Тема: Адрес переменной 

**Сложность:** легкая

**Задача:**
Напишите функцию `void change_value(float *ptr, float new_value)`, которая:
1. Выведет значение `*ptr` до изменения.
2. Изменит значение по адресу `ptr` на `new_value`.
3. Выведет значение `*ptr` после изменения.

Формат вывода:
Value changed from `old_val` to `new_val`
"""
        else:
            return """### Тема: Адрес переменной 

**Сложность:** легкая

**Задача:**
Напишите функцию `void transform_number(int *ptr, int new_value)`, которая:
1. Выведет значение `*ptr` до изменения.
2. Изменит значение по адресу `ptr` на `new_value`.
3. Выведет значение `*ptr` после изменения.

Формат вывода:
Before: `old_value`
After: `new_value`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        if self.seed % 2 == 0:
            test_cases = [
                (5.0, 10.0), (0.0, 100.0), (-5.0, 20.0), (42.0, 0.0), (-10.0, -20.0)
            ]
            for _ in range(self.tests_num - len(test_cases)):
                test_cases.append((
                    float(random.randint(-1000, 1000)),
                    float(random.randint(-1000, 1000))
                ))

            for old_val, new_val in test_cases:
                expected = f"Value changed from {old_val:g} to {new_val:g}\n"
                input_str = f"{old_val:g} {new_val:g}"

                self.tests.append(TestItem(
                    input_str=input_str,
                    showed_input=f"*ptr = {old_val:g}, new_value = {new_val:g}",
                    expected=expected,
                    compare_func=self._compare_default
                ))
        else:
            test_cases = [
                (5, 10), (0, 100), (-5, 20), (42, 0), (-10, -20)
            ]
            for _ in range(self.tests_num - len(test_cases)):
                test_cases.append((
                    random.randint(-1000, 1000),
                    random.randint(-1000, 1000)
                ))

            for old_val, new_val in test_cases:
                expected = f"Before: {old_val}\nAfter: {new_val}\n"
                input_str = f"{old_val} {new_val}"

                self.tests.append(TestItem(
                    input_str=input_str,
                    showed_input=f"*ptr = {old_val}, new_value = {new_val}",
                    expected=expected,
                    compare_func=self._compare_default
                ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if self.seed % 2 == 0:
            if "void change_value(float *ptr, float new_value)" not in self.solution:
                return "Ошибка: функция change_value имеет неверную сигнатуру или отсутствует."
        else:
            if "void transform_number(int *ptr, int new_value)" not in self.solution:
                return "Ошибка: функция transform_number имеет неверную сигнатуру или отсутствует."

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
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        
        return normalize(output) == normalize(expected)
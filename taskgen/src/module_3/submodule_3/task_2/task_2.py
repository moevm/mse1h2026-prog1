from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule3_Task2(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        variant = self.seed % 2
        size = max(2, (self.seed % 10) + 2)
        check_idx = self.seed % size
        if variant == 0:
            k_val = (self.seed % 50) + 1
            return {"func": "fill_int_via_ptr", "type": "int", "k": k_val, "size": size, "idx": check_idx, "fmt": "Value: %d\n"}
        else:
            k_val = ((self.seed % 20) + 1) * 0.5
            return {"func": "fill_float_via_ptr", "type": "float", "k": k_val, "size": size, "idx": check_idx, "fmt": "Value: %.1f\n"}

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>
#include <stdlib.h>

void {p["func"]}({p["type"]} *arr, int size, {p["type"]} k, int check_idx);

int main() {{
    int size, idx;
    double k_in;
    if (scanf("%d %lf %d", &size, &k_in, &idx) != 3) return 1;
    {p["type"]} k = ({p["type"]})k_in;
    {p["type"]} *arr = malloc(size * sizeof({p["type"]}));
    if (!arr) return 1;
    {p["func"]}(arr, size, k, idx);
    free(arr);
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        fmt_escaped = p["fmt"].replace('\n', '\\n')
        return f"""### Тема: Инициализация массива через указатель
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {p["func"]}({p["type"]} *arr, int size, {p["type"]} k, int check_idx)`, которая последовательно заполняет переданный массив в цикле по формуле `*(arr + i) = i * k`. Обработка данных должна выполняться исключительно через указательную арифметику и явное разыменование, при этом использование синтаксиса прямой индексации `arr[i]` запрещено. По завершении заполнения необходимо вывести значение элемента с индексом `check_idx`, соблюдая формат `{fmt_escaped}`.

**Формат вывода:** `{fmt_escaped}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        p = self._get_params()
        base_sizes = [2, 5, 10, 15]
        for _ in range(self.tests_num):
            size = random.choice(base_sizes) if self.tests_num > 4 else base_sizes.pop(0) if base_sizes else random.randint(2, 15)
            idx = random.randint(0, size - 1)
            k_val = p["k"] if p["type"] == "int" else ((random.randint(1, 50)) * 0.5)
            expected_val = idx * k_val
            if p["type"] == "float":
                expected_str = f"Value: {expected_val:.1f}\n"
                k_in = k_val
            else:
                expected_str = f"Value: {int(expected_val)}\n"
                k_in = int(k_val)

            self.tests.append(TestItem(
                input_str=f"{size} {k_in} {idx}",
                showed_input=f"size={size}, k={k_in}, idx={idx}",
                expected=expected_str,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\s*arr\s*,\s*int\s+size\s*,\s*{p["type"]}\s+k\s*,\s*int\s+check_idx\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура функции `{p['func']}`."

        if re.search(r'\barr\s*\[', code):
            return "Ошибка: прямая индексация arr[i] запрещена. Используйте указательную арифметику."

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
        return output.replace('\r\n', '\n').replace('\r', '\n').strip() == expected.replace('\r\n', '\n').replace('\r', '\n').strip()
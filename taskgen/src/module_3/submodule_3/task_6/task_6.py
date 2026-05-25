from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule3_Task6(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_test_driver_code(self) -> str:
        variant = self.seed % 4
        func_names = ["calc_row_sum", "find_col_max", "calc_col_sum", "find_row_min"]
        func_name = func_names[variant]
        
        return f'''#include <stdio.h>
#define COLS 4

void {func_name}(int matrix[][COLS], int rows, int target);

int main() {{
    int rows, target;
    if (scanf("%d %d", &rows, &target) != 2) return 1;
    if (rows <= 0 || rows > 10) return 1;

    int matrix[10][COLS];
    for (int i = 0; i < rows; i++) {{
        for (int j = 0; j < COLS; j++) {{
            scanf("%d", &matrix[i][j]);
        }}
    }}
    
    {func_name}(matrix, rows, target);
    return 0;
}}
'''

    def generate_task(self) -> str:
        variant = self.seed % 4
        func_names = ["calc_row_sum", "find_col_max", "calc_col_sum", "find_row_min"]
        operations = [
            "вычислить сумму target строки",
            "найти максимальный элемент в target столбце",
            "вычислить сумму target столбца",
            "найти минимальный элемент в target строке"
        ]
        formats = ["Sum row %d: %d\\n", "Max col %d: %d\\n", "Sum col %d: %d\\n", "Min row %d: %d\\n"]

        func_name = func_names[variant]
        operation = operations[variant]
        print_format = formats[variant]

        return f"""### Тема: Многомерные массивы

**Сложность:** средняя
**Задание:**
Реализуйте функцию `{func_name}(int matrix[][COLS], int rows, int target)`, которая принимает двумерный массив фиксированной ширины равной 4, количество строк `rows` и индекс `target`. Функция должна {operation}. 
Формат вывода:
{print_format}
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        
        variant = self.seed % 4
        formats = ["Sum row %d: %d\n", "Max col %d: %d\n", "Sum col %d: %d\n", "Min row %d: %d\n"]
        fmt = formats[variant]

        base_cases = [
            (3, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            (2, 0, [10, 20, 30, 40, -5, -4, -3, -2]),
            (4, 2, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]),
            (3, 3, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        ]

        for _ in range(self.tests_num - len(base_cases)):
            rows = random.randint(2, 6)
            max_target = rows if variant in [0, 3] else 4
            target = random.randint(0, max_target - 1)
            matrix = [random.randint(-50, 50) for _ in range(rows * 4)]
            base_cases.append((rows, target, matrix))

        for rows, target, matrix in base_cases:
            if variant == 0: 
                start = target * 4
                res = sum(matrix[start:start + 4])
            elif variant == 1:  
                res = max(matrix[i * 4 + target] for i in range(rows))
            elif variant == 2:  
                res = sum(matrix[i * 4 + target] for i in range(rows))
            else:             
                start = target * 4
                res = min(matrix[start:start + 4])

            expected = fmt % (target, res)
            input_str = f"{rows} {target} " + " ".join(map(str, matrix))

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"rows={rows}, target={target}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        variant = self.seed % 4
        func_names = ["calc_row_sum", "find_col_max", "calc_col_sum", "find_row_min"]
        func_name = func_names[variant]

        if f"{func_name}(" not in self.solution:
            return f"Ошибка: функция {func_name} не найдена или имеет неверное имя."

        sig_pattern = rf"{func_name}\s*\(\s*int\s+matrix\s*\[\s*\]\s*\[\s*(COLS|4)\s*\]\s*,\s*int\s+rows\s*,\s*int\s+target\s*\)"
        if not re.search(sig_pattern, self.solution):
            return "Ошибка: сигнатура функции должна быть `int matrix[][COLS], int rows, int target`."

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
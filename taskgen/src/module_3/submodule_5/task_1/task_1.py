from typing import Optional
import random
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task1(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def generate_task(self) -> str:
        type_str = "int" if self.seed % 2 == 0 else "float"
        value = self.seed if self.seed % 2 == 0 else self.seed / 2.0
        return f"""### Тема: "Функция malloc()"
**Сложность:** средняя

**Задание:**
 Реализуйте функцию `alloc_and_print(int size)`, которая внутри себя выделяет память с помощью `malloc` для массива из `size` элементов типа `{type_str}`. Инициализируйте массив значениями по формуле: `arr[i] = {value} * i`. Выведите все элементы массива в формате `Array: val1, val2, ..., valN`. Освободите выделенную память с помощью `free()`. Проверка результата `malloc` на `NULL` обязательна: если выделение памяти оказалось неудачным, выведите `Allocation failed` в `stderr` и завершите работу функции.
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        type_str = "int" if self.seed % 2 == 0 else "float"
        value = self.seed if self.seed % 2 == 0 else self.seed / 2.0

        base_sizes = [1, 3, 5, 10, 15]
        for _ in range(self.tests_num - len(base_sizes)):
            base_sizes.append(random.randint(2, 20))

        for size in base_sizes:
            input_str = str(size)
            if size <= 0:
                expected = ""
            else:
                vals = [value * i for i in range(size)]
                if type_str == "int":
                    formatted_vals = [str(int(v)) for v in vals]
                else:
                    formatted_vals = [f"{v:g}" for v in vals]
                expected = "Array: " + ", ".join(formatted_vals) + "\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"size={size}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err
        if "alloc_and_print(int size)" not in self.solution:
            return "Ошибка: функция alloc_and_print имеет неверную сигнатуру или отсутствует."
        if "malloc" not in self.solution:
            return "Ошибка: не найдено использование malloc."
        if "free" not in self.solution:
            return "Ошибка: не найдено использование free."
        if "NULL" not in self.solution and "nullptr" not in self.solution:
            return "Ошибка: не найдена проверка результата malloc на NULL."
        return None

    def _build_program_source(self) -> str:
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            "    int size;\n"
            '    if (scanf("%d", &size) != 1) return 1;\n'
            "    alloc_and_print(size);\n"
            "    return 0;\n"
            "}\n"
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
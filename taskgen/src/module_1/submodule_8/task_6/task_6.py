from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


class Module_1_Submodule_8_task_6(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_task(self) -> str:
        return (
            "Напишите программу, которая считывает с помощью `scanf` три значения: целое число `a`, символ операции `op` "
            "и целое число `b`. Программа должна вычислить результат арифметической операции `a op b` и вывести его на экран. "
            "Поддерживаемые операции: `+` (сложение), `-` (вычитание), `*` (умножение), `/` (целочисленное деление). "
            "При делении на ноль вывести строку `Division by zero` (без кавычек). Если получена неизвестная операция, выведите: "
            "`Invalid operator`. Ввод данных: числа и знак разделяются пробельными символами, знак операции не является пробелом. "
            "Вывод: одно целое число или указанная строка ошибки.\n\n"
        )

    def compile(self) -> Optional[str]:
        error = self._check_structure()
        if error:
            return error
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(self.solution, encoding="utf-8")
                comp = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    cwd=tmpdir,
                )
                if comp.returncode != 0:
                    return f"Ошибка компиляции:\n{comp.stdout.decode()}"
        except Exception as e:
            return f"Ошибка при компиляции: {e}"
        return None

    def _check_structure(self) -> Optional[str]:
        lines = []
        for line in self.solution.split('\n'):
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        clean = '\n'.join(lines)
        if not re.search(r'#include\s*<\s*stdio\.h\s*>', clean):
            return "Ошибка"
        if not re.search(r'\bmain\s*\(', clean):
            return "Ошибка"
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(input_str="5 + 3\n", showed_input="", expected="8", compare_func=self._compare_default),
            TestItem(input_str="10 - 4\n", showed_input="", expected="6", compare_func=self._compare_default),
            TestItem(input_str="6 * 7\n", showed_input="", expected="42", compare_func=self._compare_default),
            TestItem(input_str="8 / 2\n", showed_input="", expected="4", compare_func=self._compare_default),
            TestItem(input_str="5 / 0\n", showed_input="", expected="Division by zero", compare_func=self._compare_default),
            TestItem(input_str="7 % 2\n", showed_input="", expected="Invalid operator", compare_func=self._compare_default),
            TestItem(input_str="-3 + 2\n", showed_input="", expected="-1", compare_func=self._compare_default),
            TestItem(input_str="-10 / 3\n", showed_input="", expected="-3", compare_func=self._compare_default),
        ]
        self.test_extra = [{} for _ in self.tests]

    def _build_reference_program(self) -> str:
        return ""

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        error = self._check_structure()
        if error:
            return error, ""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(self.solution, encoding="utf-8")
                comp = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    cwd=tmpdir,
                )
                if comp.returncode != 0:
                    return comp.stdout.decode(), test.expected
                run_stud = subprocess.run(
                    [str(exe_path)],
                    input=test.input_str.encode(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode()
                if run_stud.returncode != 0:
                    return output, test.expected
                if output == test.expected:
                    return None
                return output, test.expected
        except Exception as e:
            return f"Ошибка выполнения: {e}", test.expected
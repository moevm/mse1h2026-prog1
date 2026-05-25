from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


class Module_1_Submodule_8_task_7(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_task(self) -> str:
        return (
            "Напишите программу, которая считывает целое число `N`, а затем строку `str`, содержащую пробелы "
            "(до символа перевода строки). Число и строка находятся на разных строках (разделены `'\\n'`). "
            "Затем выводит их через пробел в формате: `N str`. Важно: из-за буферизации после `scanf(\"%d\", &N)` "
            "во входном буфере остаётся символ `'\\n'`, не забудьте учесть это. Требования к программе:\n"
            "  - Для ввода использует стандартные функции (`scanf`, `getchar`, `fgets`).\n"
            "  - Длина строки не превышает 100 символов.\n"
            "  - Выводит ровно одно число и строку, разделенные пробелом (допустим перевод строки в конце).\n\n"
        )

    def compile(self) -> Optional[str]:
        error = self._check_includes_and_main()
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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    cwd=tmpdir,
                )
                if comp.returncode != 0:
                    return f"Ошибка компиляции:\n{comp.stdout.decode()}"
        except Exception as e:
            return f"Ошибка при компиляции: {e}"
        return None

    def _check_includes_and_main(self) -> Optional[str]:
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
            TestItem(input_str="42\nHello world\n", showed_input="", expected="42 Hello world", compare_func=self._compare_stripped),
            TestItem(input_str="0\n   spaces   \n", showed_input="", expected="0    spaces   ", compare_func=self._compare_stripped),
            TestItem(input_str="-5\nTest\n", showed_input="", expected="-5 Test", compare_func=self._compare_stripped),
            TestItem(input_str="123\n\n", showed_input="", expected="123 ", compare_func=self._compare_stripped),
            TestItem(input_str="7\n leading and trailing \n", showed_input="", expected="7  leading and trailing ", compare_func=self._compare_stripped),
        ]
        self.test_extra = [{} for _ in self.tests]

    @staticmethod
    def _compare_stripped(output: str, expected: str) -> bool:
        return output.strip() == expected.strip()

    def _build_reference_program(self) -> str:
        return ""

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        error = self._check_includes_and_main()
        if error:
            return error, ""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(self.solution, encoding="utf-8")
                comp_stud = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    cwd=tmpdir,
                )
                if comp_stud.returncode != 0:
                    return comp_stud.stdout.decode(), test.expected
                run_stud = subprocess.run(
                    [str(exe_path)],
                    input=test.input_str.encode(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode()
                if run_stud.returncode != 0:
                    return output, test.expected
                if self._compare_stripped(output, test.expected):
                    return None
                return output, test.expected
        except Exception as e:
            return f"Ошибка выполнения: {e}", test.expected
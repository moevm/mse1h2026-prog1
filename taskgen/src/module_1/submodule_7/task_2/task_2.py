from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (2, "101", 5),
    (3, "21", 7),
    (4, "13", 7),
    (5, "34", 19),
    (8, "77", 63),
    (9, "10", 9),
    (7, "123", 66),
    (2, "1111", 15),
    (6, "50", 30),
    (3, "100", 9),
    (4, "22", 10),
    (5, "11", 6),
    (8, "100", 64),
    (9, "88", 80),
    (7, "66", 48),
    (2, "10", 2),
]


class Module_1_Submodule_7_task_2(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        idx = seed_value % len(_VARIANTS)
        self.system, self.input_str, self.expected_output = _VARIANTS[idx]
        self.expected_output = str(self.expected_output)

    def generate_task(self) -> str:
        return (
            f"Напишите функцию, которая переводит число из системы счисления с основанием {self.system} "
            f"в десятичную систему счисления и возвращает результат. "
            f"Функция имеет сигнатуру: `int function(int value);`.\n"
        )

    def compile(self) -> Optional[str]:
        full_program = self._wrap_solution(self.solution)
        error = self._check_solution_content(self.solution)
        if error:
            return error

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(full_program, encoding="utf-8")
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

    def _wrap_solution(self, student_code: str) -> str:
        return (
            "#include <stdio.h>\n\n"
            f"{student_code}\n\n"
            "int main() {\n"
            "    int x;\n"
            '    scanf("%d", &x);\n'
            '    printf("%d", function(x));\n'
            "    return 0;\n"
            "}\n"
        )

    def _check_solution_content(self, code: str) -> Optional[str]:
        lines = []
        for line in code.split('\n'):
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        clean_code = '\n'.join(lines)

        if re.search(r'#include', clean_code):
            return "Ошибка"
        if re.search(r'\bmain\s*\(', clean_code):
            return "Ошибка"
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str=self.input_str + "\n",
                showed_input="",
                expected=self.expected_output,
                compare_func=self._compare_default,
            )
        ]
        self.test_extra = [{}]

    def _build_reference_program(self) -> str:
        return ""

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        error = self._check_solution_content(self.solution)
        if error:
            return error, ""

        full_program = self._wrap_solution(self.solution)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(full_program, encoding="utf-8")
                comp_stud = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    cwd=tmpdir,
                )
                if comp_stud.returncode != 0:
                    return comp_stud.stdout.decode(), self.expected_output

                run_stud = subprocess.run(
                    [str(exe_path)],
                    input=test.input_str.encode(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode().strip()
                if run_stud.returncode != 0:
                    return output, self.expected_output
                if output == self.expected_output:
                    return None
                return output, self.expected_output
        except Exception as e:
            return f"Ошибка выполнения: {e}", self.expected_output
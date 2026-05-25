from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (2, 42),
    (3, 42),
    (4, 42),
    (5, 42),
    (6, 42),
    (7, 42),
    (8, 42),
    (9, 42),
    (2, 255),
    (8, 64),
    (3, 100),
    (5, 77),
]


class Module_1_Submodule_7_task_1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        idx = seed_value % len(_VARIANTS)
        self.system, self.input_num = _VARIANTS[idx]
        self.expected_output = self._convert_to_base(self.input_num, self.system)

    def _convert_to_base(self, num: int, base: int) -> str:
        if num == 0:
            return "0"
        digits = []
        while num > 0:
            digits.append(str(num % base))
            num //= base
        return ''.join(reversed(digits))

    def generate_task(self) -> str:
        return (
            f"Напишите процедуру, которая переводит число из десятичной системы счисления в систему счисления с основанием {self.system} "
            f"и выводит результат на экран. Процедура имеет сигнатуру: `void function(int value);`.\n"
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
            "    function(x);\n"
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
                input_str=f"{self.input_num}\n",
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
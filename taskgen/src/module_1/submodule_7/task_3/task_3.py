from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (2, 16, 1),
    (2, 15, 0),
    (3, 81, 1),
    (3, 20, 0),
    (4, 64, 1),
    (4, 12, 0),
    (5, 125, 1),
    (5, 100, 0),
    (6, 1, 1),
    (7, 7, 1),
    (7, 14, 0),
    (8, 512, 1),
    (9, 729, 1),
    (9, 1000, 0),
    (2, 1, 1),
    (3, 1, 1),
]


class Module_1_Submodule_7_task_3(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        idx = seed_value % len(_VARIANTS)
        self.number, self.test_input, self.expected = _VARIANTS[idx]
        self.expected_str = str(self.expected)

    def generate_task(self) -> str:
        return (
            f"Напишите функцию для проверки, является ли полученное натуральное число целой степенью числа {self.number}. "
            f"Функция должна иметь сигнатуру: `int function(int value);`. "
            f"Если число является степенью, функция должна вернуть 1, иначе 0.\n\n"
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
            f"const int number = {self.number};\n\n"
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
                input_str=str(self.test_input) + "\n",
                showed_input="",
                expected=self.expected_str,
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
                    return comp_stud.stdout.decode(), self.expected_str

                run_stud = subprocess.run(
                    [str(exe_path)],
                    input=test.input_str.encode(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode().strip()
                if run_stud.returncode != 0:
                    return output, self.expected_str
                if output == self.expected_str:
                    return None
                return output, self.expected_str
        except Exception as e:
            return f"Ошибка выполнения: {e}", self.expected_str
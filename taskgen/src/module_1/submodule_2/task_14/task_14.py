from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (0.42, 0.44, 1e-9),
    (0.0, 0.0, 1e-12),
    (1.0, 1.0 + 1e-15, 1e-14),
    (3.14159, 3.14160, 1e-5),
    (2.718281828, 2.718281829, 1e-9),
    (1000.0, 1000.0001, 1e-4),
    (0.1 + 0.2, 0.3, 1e-15),
    (5.0, 5.000000001, 1e-9),
    (-0.5, -0.5000001, 1e-6),
    (1e-8, 2e-8, 1e-8),
]


class Module_1_Submodule_2_task_14(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.A, self.B, self.EPS = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        return (
            f"Напишите программу, которая сравнивает два вещественных числа двойной точности с заданной погрешностью.\n"
            f"Программа должна содержать `#include <stdio.h>`, `#include <math.h>` и функцию `main`.\n"
            f"Логика сравнения:\n"
            f" - переменной `a` присвоить значение `{self.A}`;\n"
            f" - переменной `b` присвоить значение `{self.B}`;\n"
            f" - переменной `eps` присвоить значение `{self.EPS}`;\n"
            f" - если `fabs(a - b) < eps`, вывести `\"yes\"`, иначе вывести `\"no\"` (без кавычек).\n"
        )

    def compile(self) -> Optional[str]:
        error = self._check_includes_and_macros()
        if error:
            return error

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(self.solution, encoding="utf-8")
                comp = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path), "-lm"],
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

    def _check_includes_and_macros(self) -> Optional[str]:
        lines = []
        for line in self.solution.split('\n'):
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        clean_code = '\n'.join(lines)

        if not re.search(r'#include\s*<\s*stdio\.h\s*>', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'#include\s*<\s*math\.h\s*>', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'\bmain\s*\(', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'\bfabs\s*\(', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if "yes" not in clean_code or "no" not in clean_code:
            return "Ошибка: программа содержит не все необходимые модули."
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected="",
                compare_func=self._compare_default,
            )
        ]
        self.test_extra = [{}]

    def _build_reference_program(self) -> str:
        return (
            "#include <stdio.h>\n"
            "#include <math.h>\n\n"
            "int main() {\n"
            f"    double a = {self.A};\n"
            f"    double b = {self.B};\n"
            f"    double eps = {self.EPS};\n"
            "    if (fabs(a - b) < eps)\n"
            "        printf(\"yes\");\n"
            "    else\n"
            "        printf(\"no\");\n"
            "    return 0;\n"
            "}\n"
        )

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        error = self._check_includes_and_macros()
        if error:
            return error, ""

        reference_src = self._build_reference_program()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                ref_src_path = tmp_path / "ref.c"
                ref_exe_path = tmp_path / "ref.x"
                ref_src_path.write_text(reference_src, encoding="utf-8")
                comp_ref = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(ref_src_path), "-o", str(ref_exe_path), "-lm"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    cwd=tmpdir,
                )
                if comp_ref.returncode != 0:
                    return f"Внутренняя ошибка: не удалось скомпилировать эталонный код:\n{comp_ref.stdout.decode()}", ""
                run_ref = subprocess.run(
                    [str(ref_exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                if run_ref.returncode != 0:
                    return f"Внутренняя ошибка: эталонная программа завершилась с ошибкой:\n{run_ref.stderr.decode()}", ""
                expected_output = run_ref.stdout.decode().strip()
        except Exception as e:
            return f"Внутренняя ошибка: {e}", ""

        student_src = self.solution
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                src_path = tmp_path / "student.c"
                exe_path = tmp_path / "student.x"
                src_path.write_text(student_src, encoding="utf-8")
                comp_stud = subprocess.run(
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path), "-lm"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    cwd=tmpdir,
                )
                if comp_stud.returncode != 0:
                    return comp_stud.stdout.decode(), expected_output
                run_stud = subprocess.run(
                    [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode().strip()
                if run_stud.returncode != 0:
                    return output, expected_output
                if output == expected_output:
                    return None
                return output, expected_output
        except Exception as e:
            return f"Ошибка выполнения: {e}", expected_output
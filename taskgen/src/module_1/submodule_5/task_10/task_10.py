from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (5, 3),
    (-2, -2),
    (0, 1),
    (10, 10),
    (-5, -10),
    (7, 9),
    (0, -1),
    (100, 99),
    (-3, -3),
    (4, 4),
]


class Module_1_Submodule_5_task_10(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.A, self.B = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        return (
            f"Напишите программу, которая выводит число, определяющее результат сравнения двух целых чисел {self.A} и {self.B}:\n"
            f"  - вывести `-1`, если {self.A} < {self.B};\n"
            f"  - вывести `0`, если {self.A} == {self.B};\n"
            f"  - вывести `1`, если {self.A} > {self.B}.\n"
            f"Ограничения:\n"
            f"  - В программе запрещено использовать условные операторы (`if`, `switch`, тернарный оператор `?:`).\n"
            f"  - Разрешается использовать только операторы сравнения и арифметические операции (в том числе логические, которые дают `0` или `1`).\n"
            f"  - Программа должна содержать `#include <stdio.h>` и функцию `main`.\n"
            f"  - Вывод осуществляется с помощью `printf`.\n"
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

    def _check_includes_and_macros(self) -> Optional[str]:
        lines = []
        for line in self.solution.split('\n'):
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        clean_code = '\n'.join(lines)

        if not re.search(r'#include\s*<\s*stdio\.h\s*>', clean_code):
            return "Ошибка"
        if not re.search(r'\bmain\s*\(', clean_code):
            return "Ошибка"

        if re.search(r'\bif\b', clean_code):
            return "Ошибка: использование оператора if запрещено."
        if re.search(r'\bswitch\b', clean_code):
            return "Ошибка: использование оператора switch запрещено."
        if re.search(r'\?', clean_code):
            if re.search(r'\?.*:', clean_code):
                return "Ошибка: использование тернарного оператора запрещено."

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
            "\n"
            "int main() {\n"
            f'    printf("%d", ({self.A} > {self.B}) - ({self.A} < {self.B}));\n'
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
                    ["gcc", "-std=c11", "-O2", str(ref_src_path), "-o", str(ref_exe_path)],
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
                    ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
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
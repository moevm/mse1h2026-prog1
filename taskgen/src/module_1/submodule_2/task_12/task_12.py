from dataclasses import dataclass
from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    type_name: str
    format_spec: str
    min_macro: str
    max_macro: str


_VARIANTS = [
    VariantSpec("float", "%e", "FLT_MIN", "FLT_MAX"),
    VariantSpec("double", "%e", "DBL_MIN", "DBL_MAX"),
    VariantSpec("long double", "%Le", "LDBL_MIN", "LDBL_MAX"),
]


class Module_1_Submodule_2_task_12(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            f"Напишите программу, которая выводит на экран минимальное и максимальное значение "
            f"вещественного типа `{v.type_name}`, используя соответствующие макросы из `<float.h>`.\n"
            f"Программа должна содержать `#include <stdio.h>`, `#include <float.h>` и функцию `main`.\n"
            f"Спецификатор формата для `printf`: `{v.format_spec}`.\n"
            f"Вывод должен состоять из двух чисел, разделенных пробелом, и заканчиваться символом перевода строки.\n"
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
        v = self.variant
        lines = []
        for line in self.solution.split('\n'):
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        clean_code = '\n'.join(lines)

        if not re.search(r'#include\s*<\s*stdio\.h\s*>', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'#include\s*<\s*float\.h\s*>', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'\b' + re.escape(v.min_macro) + r'\b', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'\b' + re.escape(v.max_macro) + r'\b', clean_code):
            return "Ошибка: программа содержит не все необходимые модули."
        if not re.search(r'\bmain\s*\(', clean_code):
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
        v = self.variant
        return (
            "#include <stdio.h>\n"
            "#include <float.h>\n\n"
            "int main() {\n"
            f'    printf("{v.format_spec} {v.format_spec}\\n", {v.min_macro}, {v.max_macro});\n'
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
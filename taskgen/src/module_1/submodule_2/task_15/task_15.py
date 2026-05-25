from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    ("0.0/0.0", "nan"),
    ("1.0/0.0", "inf"),
    ("0.0", "finite"),
    ("-1.0/0.0", "inf"),
    ("3.14", "finite"),
    ("0.0/-0.0", "nan"),
    ("1e38f*1e38f", "inf"),
    ("-0.0", "finite"),
    ("-0.0/0.0", "nan"),
    ("-1e38f*1e38f", "inf"),
]


class Module_1_Submodule_2_task_15(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.value_str, self.expected_type = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        return (
            f"Напишите программу, которая определяет, является ли значение переменной `x` "
            f"(равной `{self.value_str}`) NaN, бесконечностью или конечным числом, и выводит "
            f"соответственно `\"nan\"`, `\"inf\"` или `\"finite\"`.\n"
            f"Использовать функции `isnan`, `isinf`, `fpclassify` и другие из `<math.h>` запрещается. "
            f"Программа должна опираться только на свойства арифметических операций и сравнений языка Си.\n"
            f"Обязательно подключить `#include <stdio.h>` и реализовать функцию `main`.\n"
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
        return (
            "#include <stdio.h>\n"
            "\n"
            "int main() {\n"
            f"    float x = {self.value_str};\n"
            "    if (x != x)\n"
            "        printf(\"nan\");\n"
            "    else {\n"
            "        if (x == x + 1.0)\n"
            "            printf(\"inf\");\n"
            "        else\n"
            "            printf(\"finite\");\n"
            "    }\n"
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
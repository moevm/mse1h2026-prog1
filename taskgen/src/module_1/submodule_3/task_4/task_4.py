from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (13, "шестнадцатеричная", "0xD"),
    (8, "восьмеричная", "010"),
    (255, "шестнадцатеричная", "0xFF"),
    (64, "восьмеричная", "0100"),
    (10, "шестнадцатеричная", "0xA"),
    (15, "шестнадцатеричная", "0xF"),
    (16, "шестнадцатеричная", "0x10"),
    (7, "восьмеричная", "07"),
    (100, "шестнадцатеричная", "0x64"),
    (20, "восьмеричная", "024"),
]


class Module_1_Submodule_3_task_4(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.value, self.param, self.literal = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        return (
            f"Напишите программу, которая выводит на экран число {self.value}, используя функцию `printf` "
            f"со спецификатором `%d`. Важное условие: в исходном коде значение, передаваемое в `printf`, "
            f"должно быть записано не десятичным литералом, а литералом в {self.param} системе счисления.\n"
            f"Программа должна содержать только `#include <stdio.h>` и функцию `main`. Не допускается "
            f"использование переменных, математических операций или вызовов функций, кроме `printf`. "
            f"Используйте непосредственно литерал с правильным префиксом (`0` для восьмеричной, `0x` для шестнадцатеричной).\n"
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
            return "Ошибка: программа должна содержать только #include <stdio.h>."
        if not re.search(r'\bmain\s*\(', clean_code):
            return "Ошибка: программа должна содержать функцию main."

        if self.param == "восьмеричная":
            literal_pattern = r'0[0-7]+'
        else:
            literal_pattern = r'0[xX][0-9A-Fa-f]+'

        expected_call = rf'printf\s*\(\s*"%d"\s*,\s*{literal_pattern}\s*\)'
        if not re.search(expected_call, clean_code):
            return (
                f"Ошибка: в программе должен быть единственный вызов printf с "
                f"литералом в {self.param} системе счисления (например, {self.literal})."
            )

        if re.search(r'printf\s*\([^)]*[+\-*/]', clean_code):
            return "Ошибка: запрещено использовать математические операции в аргументах printf."

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
            f'    printf("%d", {self.literal});\n'
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
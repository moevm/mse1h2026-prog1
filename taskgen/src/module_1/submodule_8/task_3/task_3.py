from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    ("Hello world", "printf"),
    ("C programming", "puts"),
    ("42", "printf"),
    ("Test string", "puts"),
    ("Hello, World!", "printf"),
    ("12345", "puts"),
    ("printf vs puts", "printf"),
    ("done", "puts"),
    ("A", "printf"),
    ("B", "puts"),
]


class Module_1_Submodule_8_task_3(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.S, self.func_name = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        func_display = self.func_name
        return (
            f"Напишите программу, которая выводит строку `{self.S}`, используя для этого функцию `{func_display}`.\n"
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
        clean_code = '\n'.join(lines)
        if not re.search(r'#include\s*<\s*stdio\.h\s*>', clean_code):
            return "Ошибка"
        if not re.search(r'\bmain\s*\(', clean_code):
            return "Ошибка"
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

    @property
    def expected_output(self) -> str:
        if self.func_name == "printf":
            return self.S
        else:
            return self.S + "\n"

    def _build_reference_program(self) -> str:
        return (
            "#include <stdio.h>\n\n"
            "int main() {\n"
            + (
                f'    printf("%s", "{self.S}");\n'
                if self.func_name == "printf"
                else f'    puts("{self.S}");\n'
            ) +
            "    return 0;\n"
            "}\n"
        )

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        error = self._check_includes_and_main()
        if error:
            return error, ""
        expected = self.expected_output
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
                    return comp_stud.stdout.decode(), expected
                run_stud = subprocess.run(
                    [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode()
                if run_stud.returncode != 0:
                    return output, expected
                if output == expected:
                    return None
                return output, expected
        except Exception as e:
            return f"Ошибка выполнения: {e}", expected
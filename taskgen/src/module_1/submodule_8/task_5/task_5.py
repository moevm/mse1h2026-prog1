from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    ("scanf", "hello"),
    ("gets", "world"),
    ("fgets", "test"),
    ("scanf", "42"),
    ("gets", "C"),
    ("fgets", "programming"),
    ("scanf", "123abc"),
    ("gets", "printf"),
    ("fgets", "example"),
    ("scanf", "done"),
]


class Module_1_Submodule_8_task_5(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.func_name, self.test_str = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        func_display = self.func_name
        return (
            f"Напишите программу, которая считывает строку (не более 100 символов) из стандартного ввода с помощью функции `{func_display}` "
            f"и выводит эту же строку на экран. Требования:\n"
            f"  - Программа должна использовать ровно одну функцию ввода.\n"
            f"  - На стандартный вход подается строка `{self.test_str}` (одно слово, без пробелов и без символа перевода строки).\n"
            f"  - Вывести прочитанную строку с помощью `printf(\"%s\", ...)`.\n"
        )

    def _example_code(self, func: str, s: str) -> str:
        if func == "scanf":
            return (
                "#include <stdio.h>\n\n"
                "int main() {\n"
                "    char buf[100];\n"
                '    scanf("%s", buf);\n'
                '    printf("%s", buf);\n'
                "    return 0;\n"
                "}"
            )
        elif func == "gets":
            return (
                "#include <stdio.h>\n\n"
                "int main() {\n"
                "    char buf[100];\n"
                "    gets(buf);\n"
                '    printf("%s", buf);\n'
                "    return 0;\n"
                "}"
            )
        else:
            return (
                "#include <stdio.h>\n\n"
                "int main() {\n"
                "    char buf[100];\n"
                "    fgets(buf, 100, stdin);\n"
                '    printf("%s", buf);\n'
                "    return 0;\n"
                "}"
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
                    ["gcc", "-std=gnu11", "-O2", str(src_path), "-o", str(exe_path)],
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
                input_str=self.test_str,
                showed_input="",
                expected=self.test_str,
                compare_func=self._compare_default,
            )
        ]
        self.test_extra = [{}]

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
                    ["gcc", "-std=gnu11", "-O2", str(src_path), "-o", str(exe_path)],
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
                if output == test.expected:
                    return None
                return output, test.expected
        except Exception as e:
            return f"Ошибка выполнения: {e}", test.expected
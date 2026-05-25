from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (14, 16, "value", "big number %d"),
    (7, 5, "x", "large %d"),
    (0, -1, "num", "positive %d"),
    (100, 50, "count", "count %d"),
    (-3, 0, "temperature", "temp %d"),
    (8, 8, "a", "equal %d"),
    (20, 10, "size", "size %d"),
    (1, 2, "n", "small number %d"),
    (-5, -10, "val", "negative %d"),
    (42, 42, "answer", "answer %d"),
]


class Module_1_Submodule_1_task_9(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        variant = _VARIANTS[seed_value % len(_VARIANTS)]
        self.VAL = variant[0]
        self.COND = variant[1]
        self.NAME = variant[2]
        self.FMT = variant[3]
        self.expected_output = self._compute_output()

    def _compute_output(self) -> str:
        if self.VAL > self.COND:
            return self.FMT % self.VAL
        else:
            return "small"

    def generate_task(self) -> str:
        unformatted = (
            f"#include <stdio.h>\n\n"
            f"int main(){{int {self.NAME} = {self.VAL};if ({self.NAME} > {self.COND})"
            f"{{printf(\"{self.FMT}\", {self.NAME});}}else{{printf(\"small\");}}return 0;}}"
        )
        return (
            "Ниже приведен код программы, записанный без соблюдения правил форматирования. "
            "Перепишите его, строго придерживаясь указанного стиля:\n\n"
            "1. отступы - 4 пробела, табуляции не используются;\n\n"
            "2. открывающая фигурная скобка блоков - на новой строке (Allman);\n\n"
            "3. после открывающей скобки блока не должно быть пустой строки.\n\n"
            "Прочие детали (пробелы внутри скобок, длина строк и т.п.) можно не менять, "
            "если они не противоречат перечисленным требованиям.\n\n"
            f"{unformatted}\n\n"
        )

    def compile(self) -> Optional[str]:
        style_error = self._check_style()
        if style_error:
            return style_error

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

    def _check_style(self) -> Optional[str]:
        code = self.solution
        if '\t' in code:
            return "Ошибка"

        lines_no_comment = []
        for line in code.split('\n'):
            if '//' in line:
                line = line[:line.index('//')]
            lines_no_comment.append(line)
        clean = '\n'.join(lines_no_comment)

        if not re.search(r'main\s*\(\)\s*\n\s*{', clean):
            return "Ошибка"
        if re.search(r'\)\s*{', clean):
            return "Ошибка"
        if re.search(r'\}\s*else', clean) and not re.search(r'\}\s*\n\s*else', clean):
            return "Ошибка"

        brace_positions = [m.start() for m in re.finditer(r'{', clean)]
        for pos in brace_positions:
            rest = clean[pos:]
            first_line_end = rest.find('\n')
            if first_line_end == -1:
                continue
            next_line_start = first_line_end + 1
            next_line_end = rest.find('\n', next_line_start)
            if next_line_end == -1:
                next_line = rest[next_line_start:]
            else:
                next_line = rest[next_line_start:next_line_end]
            if next_line.strip() == '':
                return "Ошибка"

        lines = clean.split('\n')
        for i, line in enumerate(lines):
            stripped = line.lstrip(' ')
            if stripped == '':
                continue
            leading_spaces = len(line) - len(stripped)
            if leading_spaces % 4 != 0:
                return f"Ошибка"

        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=self.expected_output,
                compare_func=self._compare_default,
            )
        ]
        self.test_extra = [{}]

    def _build_reference_program(self) -> str:
        return (
            "#include <stdio.h>\n\n"
            "int main()\n"
            "{\n"
            f"    int {self.NAME} = {self.VAL};\n"
            f"    if ({self.NAME} > {self.COND})\n"
            "    {\n"
            f'        printf("{self.FMT}", {self.NAME});\n'
            "    }\n"
            "    else\n"
            "    {\n"
            '        printf("small");\n'
            "    }\n"
            "    return 0;\n"
            "}\n"
        )

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        style_error = self._check_style()
        if style_error:
            return style_error, ""

        student_src = self.solution
        expected = test.expected
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
                    return comp_stud.stdout.decode(), expected
                run_stud = subprocess.run(
                    [str(exe_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmpdir, check=False,
                )
                output = run_stud.stdout.decode().strip()
                if run_stud.returncode != 0:
                    return output, expected
                if output == expected:
                    return None
                return output, expected
        except Exception as e:
            return f"Ошибка выполнения: {e}", expected
from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (5, 7, 0, 3, '*', '#'),
    (4, 6, 1, 2, '+', '-'),
    (6, 5, 2, 4, 'X', 'O'),
    (3, 8, 0, 2, '@', '.'),
    (7, 4, 3, 5, 'A', 'B'),
    (5, 5, 0, 1, '#', ' '),
    (4, 4, 1, 3, '?', '!'),
    (6, 7, 2, 4, '1', '0'),
    (8, 3, 0, 2, 'M', 'W'),
    (5, 9, 1, 3, '&', '%'),
]


class Module_1_Submodule_8_task_2(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.h, self.w, self.c, self.a, self.s, self.t = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        return (
            f"Напишите программу, которая выводит на экран символьную фигуру - прямоугольную сетку высотой {self.h} и шириной {self.w}. "
            f"Для каждого элемента в позиции `(i, j)`, где `i` - номер строки (сверху вниз, начиная с 0), "
            f"а `j` - номер столбца (слева направо, начиная с 0), программа должна напечатать символ по правилу:\n"
            f"  - если `(i + j + {self.c}) % {self.a} == 0`, вывести символ `{self.s}`;\n"
            f"  - иначе если `(i + j + {self.c}) % 2 == 0`, вывести символ `{self.t}`;\n"
            f"  - иначе вывести пробел (`' '`).\n"
            f"Столбцы разделяются символом табуляции.\n"
            f"Программа должна содержать `#include <stdio.h>` и функцию `main`. "
            f"Ввод отсутствует, вывод — сформированная сетка.\n"
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

    def _generate_grid(self, h, w, c, a, s, t) -> str:
        lines = []
        for i in range(h):
            row = []
            for j in range(w):
                if (i + j + c) % a == 0:
                    row.append(s)
                elif (i + j + c) % 2 == 0:
                    row.append(t)
                else:
                    row.append(' ')
            lines.append('\t'.join(row))
        return '\n'.join(lines)

    def _build_reference_program(self) -> str:
        return (
            "#include <stdio.h>\n\n"
            "int main() {\n"
            f"    int h = {self.h}, w = {self.w}, c = {self.c}, a = {self.a};\n"
            f"    char s = '{self.s}', t = '{self.t}';\n"
            "    for (int i = 0; i < h; ++i) {\n"
            "        for (int j = 0; j < w; ++j) {\n"
            "            char ch;\n"
            "            if ((i + j + c) % a == 0) ch = s;\n"
            "            else if ((i + j + c) % 2 == 0) ch = t;\n"
            "            else ch = ' ';\n"
            '            printf("%c", ch);\n'
            "            if (j < w - 1) printf(\"\\t\");\n"
            "        }\n"
            '        printf("\\n");\n'
            "    }\n"
            "    return 0;\n"
            "}\n"
        )

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        error = self._check_includes_and_main()
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
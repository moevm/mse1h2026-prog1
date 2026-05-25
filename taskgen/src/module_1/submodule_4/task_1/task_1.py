from typing import Optional
import subprocess
import tempfile
import re
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem


_VARIANTS = [
    (5, 10),
    (3, 7),
    (0, 0),
    (-2, 8),
    (100, 200),
    (1, -1),
    (42, 58),
    (-10, -5),
    (13, 27),
    (99, 1),
]


class Module_1_Submodule_4_task_1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.X, self.Y = _VARIANTS[seed_value % len(_VARIANTS)]

    def generate_task(self) -> str:
        return (
            f"Напишите программу, которая выводит сумму двух целых чисел {self.X} и {self.Y}. Обязательные требования:\n"
            f"  - В программе должны быть две разные локальные переменные с одинаковым именем `num`. Каждая из них объявляется в своём блоке (внутри `main`). В одном блоке `num` присваивается значение {self.X}, в другом — значение {self.Y}. Блоки не должны пересекаться.\n"
            f"  - Должна быть еще одна переменная (например, `sum`), в которую накапливается сумма.\n"
            f"  - Программа должна содержать `#include <stdio.h>` и функцию `main`.\n"
            f"  - Вывод осуществляется функцией `printf` со спецификатором `%d` (только число, можно с переводом строки `\\n`).\n"
            f"  - Использовать глобальные переменные или другие имена для разных сущностей запрещено - только `num` для хранения чисел в блоках.\n"
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
        if not re.search(r'\bint\s+sum\s*=\s*0\s*;', clean_code):
            return "Ошибка"

        num_decl = re.findall(r'\bint\s+num\s*=\s*[-]?\d+\s*;', clean_code)
        if len(num_decl) != 2:
            return "Ошибка"

        sum_add = re.findall(r'sum\s*\+=\s*num\s*;', clean_code)
        if len(sum_add) != 2:
            return "Ошибка"

        if not re.search(r'printf\s*\(\s*"%d\\n"\s*,\s*sum\s*\)', clean_code):
            return "Ошибка"

        before_main = clean_code.split('main')[0]
        if re.search(r'\bint\s+num\s*=', before_main):
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

    def _build_reference_program(self) -> str:
        return (
            "#include <stdio.h>\n"
            "\n"
            "int main() {\n"
            "    int sum = 0;\n"
            "    {\n"
            f"        int num = {self.X};\n"
            "        sum += num;\n"
            "    }\n"
            "    {\n"
            f"        int num = {self.Y};\n"
            "        sum += num;\n"
            "    }\n"
            '    printf("%d\\n", sum);\n'
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
from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task11(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "safe_free_int", "type": "int", "ptr": "buffer"}
        elif rem == 1:
            return {"func": "safe_free_float", "type": "float", "ptr": "data"}
        else:
            return {"func": "safe_free_char", "type": "char", "ptr": "str"}

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Защита от Double Free
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} **{p["ptr"]})`, которая безопасно освобождает память и предотвращает двойное освобождение.
Функция должна:
1. Проверить, что `{p["ptr"]}` не равен `NULL` и не указывает на `NULL`.
2. Освободить память через `free()`.
3. Присвоить `*{p["ptr"]}` значение `NULL`, чтобы указатель в вызывающем коде стал безопасным.
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = "Before 1st: ALIVE\nAfter 1st: NULL\nAfter 2nd: NULL\n"
        self.tests = [TestItem(
            input_str="",
            showed_input="",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\*\s*{p["ptr"]}\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: сигнатура должна быть `void {p['func']}({p['type']} **{p['ptr']})`."

        null_guard = rf'if\s*\(\s*{p["ptr"]}\s*&&\s*\*{p["ptr"]}\s*\)'
        if not re.search(null_guard, code):
            return f"Ошибка: обязательно проверьте `{p['ptr']}` и `*{p['ptr']}` на NULL перед освобождением."

        if not re.search(rf'free\s*\(\s*\*\s*{p["ptr"]}\s*\)', code):
            return f"Ошибка: вызовите `free(*{p['ptr']})`."

        if not re.search(rf'\*\s*{p["ptr"]}\s*=\s*NULL', code):
            return f"Ошибка: после `free` обязательно присвойте `*{p['ptr']} = NULL`."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["type"]} *{p["ptr"]} = malloc(sizeof({p["type"]}));\n'
            f'    if (!{p["ptr"]}) return 1;\n\n'
            f'    printf("Before 1st: %s\\n", {p["ptr"]} == NULL ? "NULL" : "ALIVE");\n'
            f'    {p["func"]}(&{p["ptr"]});\n'
            f'    printf("After 1st: %s\\n", {p["ptr"]} == NULL ? "NULL" : "ALIVE");\n\n'
            f'    {p["func"]}(&{p["ptr"]});\n'
            f'    printf("After 2nd: %s\\n", {p["ptr"]} == NULL ? "NULL" : "ALIVE");\n\n'
            "    return 0;\n"
            "}\n"
        )

    def _compile_and_run(self) -> tuple[bool, str]:
        program_source = self._build_program_source()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", "-Wall", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run(
                [str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            output = "\n".join(
                part for part in (
                    run_proc.stdout.decode().strip(),
                    run_proc.stderr.decode().strip(),
                ) if part
            )
            if run_proc.returncode != 0:
                return False, output
                
            return True, output

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        ok, result = self._compile_and_run()
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected

    def _compare_default(self, output: str, expected: str) -> bool:
        norm = lambda s: s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return norm(output) == norm(expected)
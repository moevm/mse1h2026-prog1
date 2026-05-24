from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task1(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "safe_deref_int", "type": "int", 
                    "ok": "Value: ", "null": "NULL ptr: ", "fmt": "%d\n", 
                    "def": "0", "test_val": "42"}
        elif rem == 1:
            return {"func": "read_float_safe", "type": "float", 
                    "ok": "Got: ", "null": "Error: ", "fmt": "%.1f\n", 
                    "def": "0.0f", "test_val": "3.14"}
        else:
            return {"func": "get_char_checked", "type": "char", 
                    "ok": "Char: ", "null": "No data: ", "fmt": "%c\n", 
                    "def": "'\\0'", "test_val": "'Z'"}

    def generate_task(self) -> str:
        p = self._get_params()
        fmt_null = f"{p['null']}{p['fmt'].replace('\n', '\\n')}"
        fmt_ok = f"{p['ok']}{p['fmt'].replace('\n', '\\n')}"
        
        return f"""### Тема: Разыменование `NULL`
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} *ptr)`, которая безопасно разыменовывает указатель. 
- Если `ptr == NULL`, функция должна вывести `{p["null"]}` и вернуть `{p["def"]}`.
- В противном случае — вывести `{p["ok"]}` и вернуть значение `*ptr`.

**Формат вывода:**
- При `ptr == NULL`: `{fmt_null}` 
- При `ptr != NULL`: `{fmt_ok}` 
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        if p["type"] == "int":
            expected = f"{p['null']}0\n{p['ok']}42\n"
        elif p["type"] == "float":
            expected = f"{p['null']}0.0\n{p['ok']}3.1\n"
        else:
            expected = f"{p['null']}\n{p['ok']}Z\n"

        self.tests = [TestItem(
            input_str="",
            showed_input="[скрыто]",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'{p["type"]}\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\s*ptr\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}({p['type']} *ptr)`."

        if not re.search(r'if\s*\(\s*ptr\s*==\s*NULL\s*\)|if\s*\(\s*!ptr\s*\)', code):
            return "Ошибка: необходимо проверить указатель на NULL перед разыменованием."

        def_val = p["def"].strip("'")
        if not re.search(rf'return\s+(?:{re.escape(p["def"])}|0)\s*;', code):
            return f"Ошибка: при NULL необходимо вернуть значение по умолчанию ({p['def']})."

        if not re.search(r'return\s+\*ptr\s*;', code):
            return "Ошибка: необходимо вернуть разыменованное значение `*ptr`."

        if p["null"] not in code or p["ok"] not in code:
            return "Ошибка: вывод должен содержать сообщения для NULL и валидного указателя."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    // Тест 1: NULL-указатель\n'
            f'    {p["func"]}(NULL);\n\n'
            f'    // Тест 2: Валидный указатель\n'
            f'    {p["type"]} val = {p["test_val"]};\n'
            f'    {p["func"]}(&val);\n\n'
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
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
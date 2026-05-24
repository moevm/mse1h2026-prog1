from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule7_Task2(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {
                "func": "copy_ints", "src_t": "int", "dst_t": "int",
                "src_p": "src", "dst_p": "dst", "fmt": "Copy[%d]: %d -> %d\n"
            }
        elif rem == 1:
            return {
                "func": "scale_floats", "src_t": "float", "dst_t": "float",
                "src_p": "input", "dst_p": "output", "fmt": "Scaled[%d]: %.1f -> %.1f\n"
            }
        else:
            return {
                "func": "mirror_chars", "src_t": "char", "dst_t": "char",
                "src_p": "read_buf", "dst_p": "write_buf", "fmt": "Mir[%d]: '%c' -> '%c'\n"
            }

    def generate_task(self) -> str:
        p = self._get_params()
        fmt = f"{p['fmt'].replace('\n', '\\n')}"
        return f"""### Тема: const correctness
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}(const {p["src_t"]} *{p["src_p"]}, {p["dst_t"]} *{p["dst_p"]}, int size)`, которая копирует элементы из массива `{p["src_p"]}` в `{p["dst_p"]}` и выводит пары "оригинал → копия" в формате `{fmt.strip()}`.
- Исходный массив `{p["src_p"]}` должен быть доступен **только для чтения** (`const` обязателен).
- Массив `{p["dst_p"]}` должен оставаться **изменяемым** (без `const`).
- Нарушение контракта приведёт к ошибке компиляции в автотесте.

**Формат вывода:** `{fmt.strip()}` (для каждого индекса)
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        if p["src_t"] == "int":
            vals = [10, 20, 30, 40, 50]
        elif p["src_t"] == "float":
            vals = [1.5, 2.5, 3.5, 4.5, 5.5]
        else:
            vals = ['A', 'B', 'C', 'D', 'E']

        lines = []
        for i, v in enumerate(vals):
            lines.append(p["fmt"] % (i, v, v))
        expected = "".join(lines)

        self.tests = [TestItem(
            input_str="",
            showed_input=f"[скрыто]",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig_pat = rf'void\s+{p["func"]}\s*\(\s*(?:const\s+{p["src_t"]}|{p["src_t"]}\s+const)\s*\*\s*{p["src_p"]}\s*,\s*{p["dst_t"]}\s*\*\s*{p["dst_p"]}\s*,\s*int\s+size\s*\)'
        if not re.search(sig_pat, code):
            return f"Ошибка: неверная сигнатура. Требуется `const {p['src_t']} *{p['src_p']}` в первом параметре."

        fmt_escaped = re.escape(p["fmt"].replace("\n", r"\n"))
        if not re.search(rf'printf\s*\(.*{fmt_escaped}', code, re.DOTALL):
            return f"Ошибка: формат вывода должен соответствовать `{p['fmt'].strip()}`."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        if p["src_t"] == "int":
            init_vals = "{10, 20, 30, 40, 50}"
        elif p["src_t"] == "float":
            init_vals = "{1.5, 2.5, 3.5, 4.5, 5.5}"
        else:
            init_vals = "{'A', 'B', 'C', 'D', 'E'}"
        size = 5
        return (
            "#include <stdio.h>\n\n"
            f"void {p['func']}(const {p['src_t']} *{p['src_p']}, {p['dst_t']} *{p['dst_p']}, int size);\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["src_t"]} {p["src_p"]}[] = {init_vals};\n'
            f'    {p["dst_t"]} {p["dst_p"]}[{size}];\n'
            f'    {p["func"]}({p["src_p"]}, {p["dst_p"]}, {size});\n'
            "    return 0;\n"
            "}\n"
        )

    def _compile_and_run(self) -> tuple[bool, str]:
        program_source = self._build_program_source()
        p = self._get_params()
        custom_args = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wcast-qual", "-Wdiscarded-qualifiers"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            
            compile_proc = subprocess.run(
                ["gcc"] + custom_args + [str(src_path), "-o", str(exe_path)],
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
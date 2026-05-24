from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule7_Task1(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {
                "func": "demo_const_ptrs", "type": "int", "arg": "val",
                "p1": "ro_ptr", "p2": "fix_ptr", "p3": "cc_ptr",
                "fmt": "1: %d\n2: %d\n3: %d\n", 
                "val_raw": "42", "fmt_print": "%d"
            }
        elif rem == 1:
            return {
                "func": "demo_const_ptrs", "type": "float", "arg": "x",
                "p1": "f_ro", "p2": "f_fix", "p3": "f_cc",
                "fmt": "1: %.2f\n2: %.2f\n3: %.2f\n",
                "val_raw": "3.14159", "fmt_print": "%.2f"
            }
        else:
            return {
                "func": "demo_const_ptrs", "type": "char", "arg": "c",
                "p1": "c_ro", "p2": "c_fix", "p3": "c_cc",
                "fmt": "1: %c\n2: %c\n3: %c\n",
                "val_raw": "'Z'", "fmt_print": "%c"
            }

    def generate_task(self) -> str:
        p = self._get_params()
        fmt = f"{p['fmt'].replace('\n', '\\n')}"
        return f"""### Тема: const int *, int * const, const int * const
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} {p["arg"]})`, которая внутри объявляет три указателя:
1. `{p["p1"]}` — указатель на константу (данные readonly)
2. `{p["p2"]}` — константный указатель (адрес readonly)
3. `{p["p3"]}` — константный указатель на константу (всё readonly)

Все три должны инициализироваться адресом аргумента `{p["arg"]}`.
Выведите разыменованные значения в формате:
`{fmt.strip()}`

**Формат вывода:** `{fmt.strip()}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        val = p["val_raw"].strip("'") 
        if p["type"] == "float":
            val = float(val)
        elif p["type"] == "int":
            val = int(val)

        expected = f"1: {val}\n2: {val}\n3: {val}\n"
        
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

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s+{p["arg"]}\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}({p['type']} {p['arg']})`."

        regex_p1 = rf'const\s+{p["type"]}\s*\*\s*{p["p1"]}\s*=\s*&\s*{p["arg"]}|{p["type"]}\s+const\s*\*\s*{p["p1"]}\s*=\s*&\s*{p["arg"]}'
        if not re.search(regex_p1, code):
            return f"Ошибка: `{p['p1']}` должен быть указателем на константу. Ключевое const относится к данным."

        regex_p2 = rf'{p["type"]}\s*\*\s*const\s+{p["p2"]}\s*=\s*&\s*{p["arg"]}'
        if not re.search(regex_p2, code):
            return f"Ошибка: `{p['p2']}` должен быть константным указателем. Ключевое const относится к указателю."

        regex_p3 = rf'const\s+{p["type"]}\s*\*\s*const\s+{p["p3"]}\s*=\s*&\s*{p["arg"]}'
        if not re.search(regex_p3, code):
            return f"Ошибка: `{p['p3']}` должен быть константным указателем на константу. Нужно два const."

        fmt_escaped = re.escape(p["fmt"].replace("\n", r"\n"))
        if not re.search(rf'printf\s*\(.*{fmt_escaped}', code, re.DOTALL):
            return f"Ошибка: формат вывода должен соответствовать шаблону `{p['fmt'].strip()}`."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n\n"
            f"void {p['func']}({p['type']} {p['arg']});\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["type"]} {p["arg"]} = {p["val_raw"]};\n'
            f'    {p["func"]}({p["arg"]});\n'
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
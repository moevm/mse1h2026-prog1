from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task9(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        v = self.seed % 2
        if v == 0:
            return {
                "func": "compare_ptrs",
                "msg_eq": "Same address",
                "has_prefix": False
            }
        else:
            return {
                "func": "ptr_equality_check",
                "msg_eq": "Equal pointers",
                "has_prefix": True
            }

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>

void {p["func"]}(int *a, int *b);

int main() {{
    int x = 10, y = 20, z = 10;
    {p["func"]}(&x, &x);
    {p["func"]}(&x, &y);
    {p["func"]}(&y, &x);
    {p["func"]}(&x, &z);
    
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        fmt_desc = "`*a` <op> `*b`" if not p["has_prefix"] else "Result: `*a` <op> `*b`"
        return f"""### Тема: Сравнение указателей
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {p["func"]}(int *a, int *b)`, которая проверяет, указывают ли указатели на один адрес.
- Если адреса совпадают (`a == b`), выведите `{p["msg_eq"]}`.
- Если адреса разные, сравните значения `*a` и `*b` и выведите результат в формате `{fmt_desc}`, подставив оператор (`<`, `>` или `==`).

**Формат вывода:**
- При равенстве адресов: `{p["msg_eq"]}`
- При разных адресах: `{fmt_desc.replace("<op>", "<оператор>")}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        if not p["has_prefix"]:
            expected = (
                f"{p['msg_eq']}\n"
                f"10 < 20\n"
                f"20 > 10\n"
                f"10 == 10\n"
            )
        else:
            expected = (
                f"{p['msg_eq']}\n"
                f"Result: 10 < 20\n"
                f"Result: 20 > 10\n"
                f"Result: 10 == 10\n"
            )

        self.tests = [TestItem(
            input_str="",
            showed_input="4_pointer_scenarios",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*int\s*\*\s*a\s*,\s*int\s*\*\s*b\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}(int *a, int *b)`."

        if not re.search(r'\ba\s*==\s*b\b|\bb\s*==\s*a\b', code):
            return "Ошибка: необходимо проверить равенство адресов указателей (a == b)."

        val_cmp = r'\*a\s*(<|>|==|!=)\s*\*b|\*b\s*(<|>|==|!=)\s*\*a'
        if not re.search(val_cmp, code):
            return "Ошибка: необходимо сравнить значения по указателям (*a и *b)."

        return None

    def _build_program_source(self) -> str:
        return f"{self.solution}\n\n{self._get_test_driver_code()}"

    def _compile_and_run(self, test_index: int) -> tuple[bool, str]:
        program_source = self._build_program_source()
        test = self.tests[test_index]

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
                input=test.input_str.encode(),
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
        test_index = self.tests.index(test)
        ok, result = self._compile_and_run(test_index)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected

    def _compare_default(self, output: str, expected: str) -> bool:
        norm = lambda s: s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return norm(output) == norm(expected)
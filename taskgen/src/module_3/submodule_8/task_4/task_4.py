from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task4(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 2
        if rem == 0:
            return {
                "func": "float_to_int_bits", "src": "float", "dst": "int",
                "fmt": "As int: %d\\n", "test_val": "3.14f", "expected_out": "As int: 1078523331\n"
            }
        else:
            return {
                "func": "int_to_float_bits", "src": "int", "dst": "float",
                "fmt": "As float: %g\\n", "test_val": "1078523331", "expected_out": "As float: 3.14\n"
            }

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Нарушение strict aliasing
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["src"]} val)`, которая интерпретирует битовое представление числа `val` как тип `{p["dst"]}` и выводит его.
Запрещено использовать приведение типов через указатели: `*({p["dst"]}*)&val`.
Используйте безопасный способ: `memcpy` или `union`.

**Формат вывода:** `{p["fmt"].strip()}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        self.tests = [TestItem(
            input_str="",
            showed_input="[скрыто]",
            expected=p["expected_out"],
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["src"]}\s+val\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}({p['src']} val)`."

        if re.search(rf'\*\s*\(\s*\({p["dst"]}\s*\*\)\s*&\s*val\s*\)', code):
            return f"Ошибка: запрещено приведение через указатель `*({p['dst']}*)&val`."

        has_memcpy = re.search(r'memcpy\s*\(', code)
        has_union = re.search(r'\bunion\b', code) and re.search(r'\.\s*\w+\s*=', code)
        
        if not (has_memcpy or has_union):
            return "Ошибка: используйте `memcpy` или `union` для безопасной реинтерпретации битов."

        fmt_escaped = re.escape(p["fmt"].replace("\n", r"\n"))
        if not re.search(rf'printf\s*\(.*{fmt_escaped}', code, re.DOTALL):
            return f"Ошибка: вывод должен соответствовать формату `{p['fmt'].strip()}`."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n"
            "#include <string.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["func"]}({p["test_val"]});\n'
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
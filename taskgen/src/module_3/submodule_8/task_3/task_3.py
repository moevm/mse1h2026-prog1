from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task3(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "create_zero_array", "type": "int", "init": "0"}
        elif rem == 1:
            return {"func": "init_float_array", "type": "float", "init": "0.0f"}
        else:
            return {"func": "alloc_cleared_buf", "type": "char", "init": "0"}

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Использование неинициализированной памяти
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{p["func"]}(int size)`, которая выделяет память для массива из `size` элементов типа `{p["type"]}` и гарантирует, что все элементы равны `{p["init"]}` до начала использования. Верните указатель на выделенную память. 
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str="",
            showed_input="",
            expected="Status: OK\n",
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'{p["type"]}\s*\*\s*{p["func"]}\s*\(\s*int\s+size\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}(int size)`."

        if not re.search(r'calloc\s*\(', code):
            return "Ошибка: используйте `calloc` для гарантированной инициализации нулём. `malloc` оставляет память неинициализированной."

        if not re.search(r'return\s+\w+\s*;', code):
            return "Ошибка: функция должна возвращать указатель на выделенную память."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    int size = 10;\n'
            f'    {p["type"]} *arr = {p["func"]}(size);\n\n'
            f'    if (!arr) {{\n'
            f'        printf("Status: FAIL (NULL)\\n");\n'
            f'        return 1;\n'
            f'    }}\n\n'
            f'    int ok = 1;\n'
            f'    for (int i = 0; i < size; i++) {{\n'
            f'        if (arr[i] != 0) {{\n'
            f'            ok = 0;\n'
            f'            break;\n'
            f'        }}\n'
            f'    }}\n\n'
            f'    if (ok) printf("Status: OK\\n");\n'
            f'    else printf("Status: FAIL\\n");\n\n'
            f'    free(arr);\n'
            f'    return 0;\n'
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
from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task12(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "safe_int_ptr", "type": "int", "val": "val", "fmt": "%d", "def": "42"}
        elif rem == 1:
            return {"func": "safe_float_ptr", "type": "float", "val": "x", "fmt": "%.2f", "def": "3.14f"}
        else:
            return {"func": "safe_char_ptr", "type": "char", "val": "c", "fmt": "%c", "def": "'A'"}

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Висячий указатель
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} {p["val"]})`, которая возвращает указатель на переданное значение `{p["val"]}`.
Запрещено возвращать адрес локальной переменной (`return &{p["val"]};`) — это создаст висячий указатель.
Обязательно используйте динамическое выделение памяти (`malloc`/`calloc`), скопируйте значение и верните указатель.
При ошибке выделения верните `NULL`.
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        if p["type"] == "int":
            expected = p["fmt"] % 42
        elif p["type"] == "float":
            expected = p["fmt"] % 3.14
        else:
            expected = p["fmt"] % 'A'

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

        sig = rf'{p["type"]}\s*\*\s+{p["func"]}\s*\(\s*{p["type"]}\s+{p["val"]}\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['type']} *{p['func']}({p['type']} {p['val']})`."

        if re.search(rf'return\s+&\s*{p["val"]}\s*;', code):
            return f"Ошибка: запрещено возвращать адрес параметра `&{p['val']}`."

        if not re.search(r'malloc\s*\(|calloc\s*\(', code):
            return "Ошибка: необходимо использовать динамическое выделение памяти (malloc/calloc)."

        alloc_match = re.search(r'(\w+)\s*=\s*(?:malloc|calloc)\s*\(', code)
        if alloc_match:
            ptr_var = alloc_match.group(1)
            null_checks = [
                rf'if\s*\(\s*!{ptr_var}\s*\)',           
                rf'if\s*\(\s*{ptr_var}\s*\)',            
                rf'if\s*\(\s*{ptr_var}\s*==\s*NULL\s*\)',
                rf'if\s*\(\s*NULL\s*==\s*{ptr_var}\s*\)',
                rf'if\s*\(\s*{ptr_var}\s*!=\s*NULL\s*\)',
                rf'if\s*\(\s*NULL\s*!=\s*{ptr_var}\s*\)',
            ]
            if not any(re.search(pat, code) for pat in null_checks):
                return f"Ошибка: необходимо проверить переменную `{ptr_var}` (результат malloc) на NULL."

        if not re.search(rf'\*\s*\w+\s*=\s*{p["val"]}\b', code):
            return f"Ошибка: значение `{p['val']}` должно быть скопировано в выделенную память."

        if not re.search(r'return\s+\w+\s*;', code):
            return "Ошибка: функция должна возвращать указатель."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["type"]} test_val = {p["def"]};\n'
            f'    {p["type"]} *ptr = {p["func"]}(test_val);\n'
            f'    if (ptr) {{\n'
            f'        printf("{p["fmt"]}", *ptr);\n'
            f'        free(ptr);\n'
            f'    }} else {{\n'
            f'        printf("NULL\\n");\n'
            f'    }}\n'
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
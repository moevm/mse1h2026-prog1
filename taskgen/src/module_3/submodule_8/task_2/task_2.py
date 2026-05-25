from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule8_Task2(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 2
        if rem == 0:
            return {"func": "safe_get_int", "type": "int", 
                    "ok_msg": "OK: ", "oom_msg": "OOB: ", "fmt": "%d\n", 
                    "default": "-1", "arr_init": "{10, 20, 30}", "valid_idx": 1, "valid_val": "20"}
        else:
            return {"func": "safe_get_float", "type": "float", 
                    "ok_msg": "Value: ", "oom_msg": "Error: ", "fmt": "%.2f\n", 
                    "default": "0.0f", "arr_init": "{1.1, 2.2, 3.3}", "valid_idx": 2, "valid_val": "3.30"}

    def generate_task(self) -> str:
        p = self._get_params()
        fmt_display = p['fmt'].replace('\n', '\\n')
        fmt_ok = f"{p['ok_msg']}{fmt_display}"
        fmt_oom = f"{p['oom_msg']}{fmt_display}"
        
        return f"""### Тема: Выход за границы
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {p["func"]}({p["type"]} *arr, int size, int index)`, которая безопасно обращается к элементу массива.
- Если `index` выходит за допустимые пределы `[0, size)`, функция должна вывести `{p["oom_msg"]}`.
- В противном случае — вывести `{p["ok_msg"]}` и значение `arr[index]`.

**Формат вывода:**
- При успехе: `{fmt_ok}`
- При ошибке: `{fmt_oom}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        
        if p["type"] == "float":
            expected = f"{p['oom_msg']}0.00\n{p['oom_msg']}0.00\n{p['ok_msg']}3.30\n"
        else:
            expected = f"{p['oom_msg']}-1\n{p['oom_msg']}-1\n{p['ok_msg']}20\n"

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

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\s*arr\s*,\s*int\s+size\s*,\s*int\s+index\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `void {p['func']}({p['type']} *arr, int size, int index)`."

        if not re.search(r'index\s*<\s*0\s*\|\||index\s*>=\s*size', code):
            return "Ошибка: необходимо проверить условие `index < 0 || index >= size`."

        if not re.search(rf'\barr\s*\[\s*index\s*\]', code):
            return "Ошибка: необходимо обращаться к элементу через `arr[index]`."

        if not re.search(rf'printf\s*\(.*{p["oom_msg"]}', code):
            return f"Ошибка: вывод должен содержать `{p['oom_msg']}` при выходе за границы."
        if not re.search(rf'printf\s*\(.*{p["ok_msg"]}', code):
            return f"Ошибка: вывод должен содержать `{p['ok_msg']}` при успешном доступе."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["type"]} arr[] = {p["arr_init"]};\n'
            f'    int size = 3;\n\n'
            f'    {p["func"]}(arr, size, -1);\n'
            f'    {p["func"]}(arr, size, 5);\n'
            f'    {p["func"]}(arr, size, {p["valid_idx"]});\n\n'
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
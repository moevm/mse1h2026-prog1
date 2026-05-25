from typing import Optional
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule6_Task2(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_params(self) -> dict:
        rem = self.seed % 3
        if rem == 0:
            return {"func": "apply_transform", "type": "int", "cb": "op", "fmt": "Result[%d]: %d\n"}
        elif rem == 1:
            return {"func": "process_values", "type": "float", "cb": "transform", "fmt": "Out[%d]: %.1f\n"}
        else:
            return {"func": "map_array", "type": "int", "cb": "mapper", "fmt": "Val[%d] => %d\n"}

    def generate_task(self) -> str:
        p = self._get_params()
        fmt = f"{p['fmt'].replace('\n', '\\n')}"
        return f"""### Тема: Передача функции как аргумент
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func"]}({p["type"]} *arr, int size, {p["type"]} (*{p["cb"]})({p["type"]}))`, которая применяет функцию-колбэк `{p["cb"]}` к каждому элементу массива `arr` и выводит результат в формате `{fmt.strip()}`.

**Формат вывода:** `{fmt.strip()}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        p = self._get_params()
        vals = [2, 4, 6, 8, 10]
        lines = []
        for i, v in enumerate(vals):
            if p["type"] == "float":
                lines.append(p["fmt"] % (i, float(v)))
            else:
                lines.append(p["fmt"] % (i, v))
        expected = "".join(lines)

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
        sig_pat = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\s*arr\s*,\s*int\s+size\s*,\s*{p["type"]}\s*\(\s*\*\s*{p["cb"]}\s*\)\s*\(\s*{p["type"]}\s*\)\s*\)'
        if not re.search(sig_pat, code):
            return f"Ошибка: неверная сигнатура функции `{p['func']}`."

        if not re.search(rf'{p["cb"]}\s*\(', code):
            return f"Ошибка: функция должна вызывать переданный колбэк `{p['cb']}`."

        return None

    def _build_program_source(self) -> str:
        p = self._get_params()
        return (
            "#include <stdio.h>\n\n"
            f"void {p['func']}({p['type']} *arr, int size, {p['type']} (*{p['cb']})({p['type']}));\n\n"
            f"{p['type']} test_callback({p['type']} x) {{\n"
            f"    return x * 2;\n"
            f"}}\n\n"
            f"{self.solution}\n\n"
            "int main(void) {\n"
            f'    {p["type"]} data[] = {{1, 2, 3, 4, 5}};\n'
            f'    {p["func"]}(data, 5, test_callback);\n'
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
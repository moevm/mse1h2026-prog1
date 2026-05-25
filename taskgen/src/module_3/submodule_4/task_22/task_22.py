from typing import Optional
import random
import re
import subprocess
import tempfile
from pathlib import Path
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task22(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {}

    def _get_test_driver_code(self) -> str:
        func_name = "bounded_strcpy" if self.seed % 2 == 0 else "str_copy_safe"
        return f'''#include <stdio.h>
#include <stddef.h>

size_t {func_name}(char *dest, size_t dest_size, const char *src);

int main() {{
    size_t dest_size;
    char src[512];
    if (scanf("%zu %511s", &dest_size, src) != 2) return 1;

    char dest[512];
    {func_name}(dest, dest_size, src);
    return 0;
}}
'''

    def generate_task(self) -> str:
        is_v0 = self.seed % 2 == 0
        func_name = "bounded_strcpy" if is_v0 else "str_copy_safe"
        label_ok = "OK" if is_v0 else "FULL"
        label_cut = "TRUNC" if is_v0 else "CUT"
        print_format = "Status: {{count}} copied | Data: {{data}} | Flag: {{flag}}" if is_v0 else "Result: len={{count}} content={{data}} check={{flag}}"

        return f"""### Тема: Опасность переполнения буфера
**Сложность:** средняя

**Задание:**
Реализуйте функцию `size_t {func_name}(char *dest, size_t dest_size, const char *src)`. Функция должна безопасно скопировать строку `src` в буфер `dest`, гарантируя, что будет записано не более `dest_size - 1` символов, а в конце буфера всегда будет стоять завершающий `'\\0'`. Запрещено использовать стандартные строковые функции (`strcpy`, `strncpy`, `memcpy`, `snprintf` и т.д.). Реализация должна корректно обрабатывать случай `dest_size == 0` или `dest_size == 1`. Внутри функции необходимо определить статус операции: `{label_ok}` - строка поместилась целиком, `{label_cut}` - строка была обрезана, и вывести результат в строгом формате:
`{print_format}`
"""

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        is_v0 = self.seed % 2 == 0
        label_ok = "OK" if is_v0 else "FULL"
        label_cut = "TRUNC" if is_v0 else "CUT"

        base_cases = [
            (10, "Hello"),       
            (6, "Hello"),        
            (5, "Hello"),        
            (3, "HelloWorld"),  
            (1, "Test"),         
            (2, "A"),            
            (15, "abc"),        
            (4, "short"),       
        ]

        while len(base_cases) < self.tests_num:
            size = random.randint(1, 12)
            s_len = random.randint(1, 15)
            src = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=s_len))
            base_cases.append((size, src))

        for dest_size, src in base_cases[:self.tests_num]:
            input_str = f"{dest_size} {src}"

            copied_len = min(dest_size - 1, len(src))
            if dest_size == 0: copied_len = 0
            
            copied_data = src[:copied_len]
            flag = label_ok if len(src) < dest_size else label_cut

            if is_v0:
                expected = f"Status: {copied_len} copied | Data: {copied_data} | Flag: {flag}\n"
            else:
                expected = f"Result: len={copied_len} content={copied_data} check={flag}\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"dest_size={dest_size}, src=\"{src}\"",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func_name = "bounded_strcpy" if self.seed % 2 == 0 else "str_copy_safe"

        sig_pattern = rf'size_t\s+{func_name}\s*\(\s*char\s*\*\s*dest\s*,\s*size_t\s+dest_size\s*,\s*const\s+char\s*\*\s*src\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: неверная сигнатура функции {func_name}."

        forbidden = ['strcpy', 'strncpy', 'memcpy', 'memmove', 'snprintf', 'sprintf', 'string.h', 'stdlib.h']
        for f in forbidden:
            if re.search(rf'\b{f}\b', self.solution):
                return f"Ошибка: запрещено использовать {f}."

        if "'\\0'" not in self.solution and "0" not in self.solution:
            return "Ошибка: не найдено явное добавление завершающего '\\0'."

        return None

    def _build_program_source(self) -> str:
        return (
            f"{self.solution}\n\n"
            f"{self._get_test_driver_code()}\n"
        )

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
        def normalize(s: str) -> str:
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
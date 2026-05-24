from dataclasses import dataclass
from typing import Optional, List
import subprocess
import tempfile
import textwrap
import re
from pathlib import Path

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    elem_type: str
    count: int
    signature: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(elem_type="int",    count=5, signature="int read_items(const char *filename, int *buf, int count)"),
    1: VariantSpec(elem_type="double", count=3, signature="int read_items(const char *filename, double *buf, int count)"),
    2: VariantSpec(elem_type="float",  count=8, signature="int read_items(const char *filename, float *buf, int count)"),
    3: VariantSpec(elem_type="short",  count=6, signature="int read_items(const char *filename, short *buf, int count)"),
}


class FReadTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.test_extra = []

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# stdlib: чтение файла\n\n"
            "### Задание №4\n\n"
            "- **Формулировка:**  \n"
            "  Вам дана сигнатура функции (соответствует вашему варианту).  \n"
            "  Напишите функцию `read_items`, которая:\n"
            "  1. Открывает файл с именем `filename` в бинарном режиме для чтения.\n"
            "  2. Считывает не более `count` элементов в массив `buf`.\n"
            "  3. Закрывает файл.\n"
            "  4. Возвращает фактическое количество прочитанных элементов.\n"
            "  Если файл не удалось открыть — возвращает `0`.\n\n"
            "  Писать `main()` не нужно.\n\n"
            f"**Сигнатура:** `{v.signature}`\n\n"
            "Подключать заголовочные файлы не требуется.\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _check_solution_text(self) -> bool:
        """Проверяет, что в решении есть вызов fclose."""
        lines = self.solution.split('\n')
        code_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            code_lines.append(line)
        code = ' '.join(code_lines)
        pattern = r'fclose\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)\s*;'
        return bool(re.search(pattern, code))

    def _check_fopen_mode(self) -> bool:
        """Проверяет, что fopen вызывается с режимом "rb" (бинарное чтение)."""
        lines = self.solution.split('\n')
        code_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            code_lines.append(line)
        code = ' '.join(code_lines)
        pattern = r'fopen\s*\([^,)]*,\s*"rb"\s*\)'
        return bool(re.search(pattern, code))

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        test_cases = [
            ("exact.bin", v.count, "ровно count элементов"),
            ("less.bin",  v.count - 1, f"{v.count-1} элементов (меньше count)"),
            ("more.bin",  v.count + 2, f"{v.count+2} элементов (больше count)"),
            ("nonexist.bin", 0, "файл не существует"),
        ]

        for filename, num_elements, description in test_cases:
            if num_elements < 0:
                continue
            expected_ret = min(num_elements, v.count) if num_elements >= 0 else 0
            if filename == "nonexist.bin":
                expected_ret = 0

            test = TestItem(
                input_str=f"filename={filename}, count={v.count}, {description}",
                showed_input=f"read_items(\"{filename}\", buf, {v.count})",
                expected=f"RET={expected_ret}",
                compare_func=self._compare_default,
            )
            self.tests.append(test)
            self.test_extra.append({
                "filename": filename,
                "num_elements": num_elements,
                "expected_ret": expected_ret
            })

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        filename = extra["filename"]
        num_elements = extra["num_elements"]
        expected_ret = extra["expected_ret"]

        data_definition = ""
        if filename != "nonexist.bin" and num_elements > 0:
            if v.elem_type == "int":
                data_list = [str(i+1) for i in range(num_elements)]
                data_decl = f"int test_data[{num_elements}] = {{{', '.join(data_list)}}};"
                write_code = f"fwrite(test_data, sizeof(int), {num_elements}, f);"
            elif v.elem_type == "double":
                data_list = [str(i+1.0) for i in range(num_elements)]
                data_decl = f"double test_data[{num_elements}] = {{{', '.join(data_list)}}};"
                write_code = f"fwrite(test_data, sizeof(double), {num_elements}, f);"
            elif v.elem_type == "float":
                data_list = [str(i+1.0) + "f" for i in range(num_elements)]
                data_decl = f"float test_data[{num_elements}] = {{{', '.join(data_list)}}};"
                write_code = f"fwrite(test_data, sizeof(float), {num_elements}, f);"
            elif v.elem_type == "short":
                data_list = [str(i+1) for i in range(num_elements)]
                data_decl = f"short test_data[{num_elements}] = {{{', '.join(data_list)}}};"
                write_code = f"fwrite(test_data, sizeof(short), {num_elements}, f);"
            else:
                raise ValueError(f"Unknown type {v.elem_type}")
            data_definition = f"""
    // Создание файла с тестовыми данными
    {data_decl}
    FILE *f = fopen("{filename}", "wb");
    if (!f) {{ printf("FAIL: cannot create test file\\n"); return 1; }}
    {write_code}
    fclose(f);
"""

        if filename == "nonexist.bin":
            check_code = f"""
    // Файл не существует, ожидаем 0
    {v.elem_type} buf[{v.count}];
    int ret = read_items("{filename}", buf, {v.count});
    printf("RET=%d\\n", ret);
    return (ret == 0) ? 0 : 1;
"""
            data_definition = ""
        else:
            check_code = f"""
    // Вызов функции студента
    {v.elem_type} buf[{v.count}];
    int ret = read_items("{filename}", buf, {v.count});

    // Проверка возвращаемого значения
    if (ret != {expected_ret}) {{
        printf("RET=%d (expected {expected_ret})\\n", ret);
        return 1;
    }}

    // Проверка содержимого буфера (только первые ret элементов)
    for (int i = 0; i < ret; i++) {{
        {v.elem_type} expected = test_data[i];
        if (buf[i] != expected) {{
            printf("RET={expected_ret} but buf[%d] mismatch\\n", i);
            return 1;
        }}
    }}
    printf("RET={expected_ret}\\n");
    return 0;
"""

        program = textwrap.dedent(f"""
        #include <stdio.h>
        #include <stdlib.h>

        {self.solution}

        int main() {{
            {data_definition}
            {check_code}
        }}
        """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if not self._check_solution_text():
            return "Ошибка: не найден вызов fclose() для закрытия файла", "требуется fclose"
        if not self._check_fopen_mode():
            return "Ошибка: файл должен открываться в бинарном режиме (\"rb\")", "требуется режим \"rb\""

        try:
            idx = self.tests.index(test)
        except ValueError:
            return "Test not found", "unknown"
        extra = self.test_extra[idx]

        program_source = self._build_test_program(extra)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "test_program.c"
            exe_path = tmp_path / "test_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                cwd=tmpdir,
            )
            if compile_proc.returncode != 0:
                return compile_proc.stdout.decode(), test.expected

            run_proc = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, check=False,
            )
            output = "\n".join(part for part in (run_proc.stdout.decode().strip(), run_proc.stderr.decode().strip()) if part)
            if run_proc.returncode != 0:
                return output, test.expected

            if output.startswith(f"RET={test.expected.split('=')[1]}"):
                return None
            return output, test.expected
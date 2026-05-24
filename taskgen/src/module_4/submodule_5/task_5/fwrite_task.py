from dataclasses import dataclass
from typing import Optional
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
    0: VariantSpec(elem_type="int",    count=5, signature="int write_items(const char *filename, const int *buf, int count)"),
    1: VariantSpec(elem_type="double", count=3, signature="int write_items(const char *filename, const double *buf, int count)"),
    2: VariantSpec(elem_type="float",  count=8, signature="int write_items(const char *filename, const float *buf, int count)"),
    3: VariantSpec(elem_type="short",  count=6, signature="int write_items(const char *filename, const short *buf, int count)"),
}


class FWriteTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.test_extra = []

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# stdlib: запись в файл\n\n"
            "### Задание №5\n\n"
            "- **Формулировка:**  \n"
            "  Вам дана сигнатура функции (соответствует вашему варианту).  \n"
            "  Напишите функцию `write_items`, которая:\n"
            "  1. Открывает файл с именем `filename` в бинарном режиме для записи.\n"
            "  2. Записывает `count` элементов из массива `buf` в файл.\n"
            "  3. Закрывает файл.\n"
            "  4. Возвращает фактическое количество записанных элементов.\n"
            "  Если файл не удалось открыть — возвращает `0`.\n\n"
            "  Писать `main()` не нужно.\n\n"
            f"**Сигнатура:** `{v.signature}`\n\n"
            "Подключать заголовочные файлы не требуется.\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _check_solution_text(self) -> bool:
        """Проверяет наличие вызова fclose."""
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
        """Проверяет, что fopen вызывается с режимом "wb" (бинарная запись)."""
        lines = self.solution.split('\n')
        code_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            code_lines.append(line)
        code = ' '.join(code_lines)
        pattern = r'fopen\s*\([^,)]*,\s*"wb"\s*\)'
        return bool(re.search(pattern, code))

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        filename1 = "write.bin"
        test1 = TestItem(
            input_str=f"filename={filename1}, count={v.count}",
            showed_input=f"write_items(\"{filename1}\", buf, {v.count})",
            expected="OK",
            compare_func=self._compare_default,
        )
        self.tests.append(test1)
        self.test_extra.append({
            "filename": filename1,
            "should_succeed": True,
            "expected_written": v.count,
        })

        filename2 = ""
        test2 = TestItem(
            input_str=f"filename='' (пустая строка), count={v.count}",
            showed_input=f"write_items(\"\", buf, {v.count})",
            expected="OK", 
            compare_func=self._compare_default,
        )
        self.tests.append(test2)
        self.test_extra.append({
            "filename": filename2,
            "should_succeed": False,
            "expected_written": 0,
        })

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        filename = extra["filename"]
        should_succeed = extra["should_succeed"]
        expected_written = extra["expected_written"]

        data_list = []
        for i in range(v.count):
            if v.elem_type == "int":
                data_list.append(str(i+1))
            elif v.elem_type == "double":
                data_list.append(str(i+1.0))
            elif v.elem_type == "float":
                data_list.append(str(i+1.0) + "f")
            elif v.elem_type == "short":
                data_list.append(str(i+1))
        data_decl = f"{v.elem_type} test_data[{v.count}] = {{{', '.join(data_list)}}};"

        if should_succeed:
            verify_code = f"""
    FILE *fr = fopen("{filename}", "rb");
    if (!fr) {{
        printf("FAIL: cannot open written file for verification\\n");
        return 1;
    }}
    {v.elem_type} read_buf[{v.count}];
    size_t nread = fread(read_buf, sizeof({v.elem_type}), {v.count}, fr);
    fclose(fr);
    if (nread != {v.count}) {{
        printf("FAIL: read %zu elements, expected {v.count}\\n", nread);
        return 1;
    }}
    for (int i = 0; i < {v.count}; i++) {{
        if (read_buf[i] != test_data[i]) {{
            printf("FAIL: mismatch at index %d\\n", i);
            return 1;
        }}
    }}
    printf("OK\\n");
    return 0;
"""
        else:
            verify_code = f"""
    printf("OK\\n");
    return (ret == 0) ? 0 : 1;
"""

        program = textwrap.dedent(f"""
        #include <stdio.h>
        #include <stdlib.h>

        {self.solution}

        int main() {{
            // Тестовые данные
            {data_decl}

            int ret = write_items("{filename}", test_data, {v.count});

            if (ret != {expected_written}) {{
                printf("FAIL: returned %d, expected {expected_written}\\n", ret);
                return 1;
            }}

            {verify_code}
        }}
        """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if not self._check_solution_text():
            return "Ошибка: не найден вызов fclose() для закрытия файла", "требуется fclose"
        if not self._check_fopen_mode():
            return "Ошибка: файл должен открываться в бинарном режиме записи (\"wb\")", "требуется режим \"wb\""

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

            if output == "OK":
                return None
            return output, test.expected
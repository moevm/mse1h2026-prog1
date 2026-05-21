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
    value_type: str
    signature_write: str
    signature_read: str
    test_values: list      # строковые представления для вставки в код


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        value_type="int",
        signature_write="void write_bin(const char *filename, int value)",
        signature_read="int read_bin(const char *filename)",
        test_values=["42", "-123", "0"],
    ),
    1: VariantSpec(
        value_type="double",
        signature_write="void write_bin(const char *filename, double value)",
        signature_read="double read_bin(const char *filename)",
        test_values=["3.141592", "-2.5", "0.0"],
    ),
    2: VariantSpec(
        value_type="long",
        signature_write="void write_bin(const char *filename, long value)",
        signature_read="long read_bin(const char *filename)",
        test_values=["123456789L", "-987654321L", "0L"],
    ),
    3: VariantSpec(
        value_type="float",
        signature_write="void write_bin(const char *filename, float value)",
        signature_read="float read_bin(const char *filename)",
        test_values=["1.2345f", "-5.6789f", "0.0f"],
    ),
}


class BinaryIOTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.test_extra = []

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# stdlib: бинарный режим\n\n"
            "### Задание №6.2\n\n"
            "- **Формулировка:**  \n"
            "  Напишите **две функции**:\n\n"
            "  1. `void write_bin(const char *filename, value_type value)` — открывает файл\n"
            "     `filename` в бинарном режиме для записи (`\"wb\"`), записывает одно значение\n"
            "     через `fwrite`, закрывает файл.\n\n"
            "  2. `value_type read_bin(const char *filename)` — открывает файл `filename`\n"
            "     в бинарном режиме для чтения (`\"rb\"`), считывает одно значение через\n"
            "     `fread`, закрывает файл и возвращает считанное значение.\n\n"
            "  Если файл не удалось открыть — функция чтения возвращает `0` (или `0.0`).\n"
            "  Писать `main()` не нужно.\n\n"
            f"**Ваш вариант:** `value_type = {v.value_type}`\n\n"
            f"Сигнатуры:\n"
            f"- `{v.signature_write}`\n"
            f"- `{v.signature_read}`\n\n"
            "Подключать заголовочные файлы не требуется.\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _check_solution_fclose(self) -> bool:
        """Проверяет наличие fclose в решении (хотя бы одного вызова)."""
        lines = self.solution.split('\n')
        code_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            code_lines.append(line)
        code = ' '.join(code_lines)
        pattern = r'fclose\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)\s*;'
        return bool(re.search(pattern, code))

    def _check_fopen_modes(self) -> bool:
        """Проверяет, что fopen используется с "wb" и "rb"."""
        lines = self.solution.split('\n')
        code_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            code_lines.append(line)
        code = ' '.join(code_lines)
        has_wb = re.search(r'fopen\s*\([^,)]*,\s*"wb"\s*\)', code) is not None
        has_rb = re.search(r'fopen\s*\([^,)]*,\s*"rb"\s*\)', code) is not None
        return has_wb and has_rb

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        for i, val_str in enumerate(v.test_values):
            filename = f"test_{i}.bin"
            test = TestItem(
                input_str=f"filename={filename}, value={val_str}",
                showed_input=f"write_bin(\"{filename}\", {val_str}); read_bin(\"{filename}\")",
                expected=f"OK:{val_str}",
                compare_func=self._compare_default,
            )
            self.tests.append(test)
            self.test_extra.append({
                "filename": filename,
                "value_str": val_str,
                "should_succeed": True,
            })

        filename = "nonexist.bin"
        test = TestItem(
            input_str=f"filename={filename} (не существует)",
            showed_input=f"read_bin(\"{filename}\")",
            expected="OK:0",
            compare_func=self._compare_default,
        )
        self.tests.append(test)
        self.test_extra.append({
            "filename": filename,
            "value_str": "0",
            "should_succeed": False,
        })

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        filename = extra["filename"]
        value_str = extra["value_str"]
        should_succeed = extra["should_succeed"]

        if should_succeed:
            clean_value = value_str.rstrip('Lf')
            if v.value_type == "double":
                compare_code = f"""
    double read_val = read_bin("{filename}");
    if (fabs(read_val - {clean_value}) > 1e-9) {{
        printf("FAIL: read %f, expected {clean_value}\\n", read_val);
        return 1;
    }}
    printf("OK:{value_str}\\n");
    return 0;
"""
            elif v.value_type == "float":
                compare_code = f"""
    float read_val = read_bin("{filename}");
    if (fabsf(read_val - {clean_value}f) > 1e-6f) {{
        printf("FAIL: read %f, expected {clean_value}\\n", read_val);
        return 1;
    }}
    printf("OK:{value_str}\\n");
    return 0;
"""
            elif v.value_type == "long":
                compare_code = f"""
    long read_val = read_bin("{filename}");
    if (read_val != {clean_value}) {{
        printf("FAIL: read %ld, expected {clean_value}\\n", read_val);
        return 1;
    }}
    printf("OK:{value_str}\\n");
    return 0;
"""
            else:  # int
                compare_code = f"""
    int read_val = read_bin("{filename}");
    if (read_val != {clean_value}) {{
        printf("FAIL: read %d, expected {clean_value}\\n", read_val);
        return 1;
    }}
    printf("OK:{value_str}\\n");
    return 0;
"""
            program = textwrap.dedent(f"""
            #include <stdio.h>
            #include <math.h>

            {self.solution}

            int main() {{
                write_bin("{filename}", {value_str});
                {compare_code}
            }}
            """)
        else:
            if v.value_type == "double":
                compare_code = f"""
    double read_val = read_bin("{filename}");
    if (read_val != 0.0) {{
        printf("FAIL: read %f, expected 0.0\\n", read_val);
        return 1;
    }}
    printf("OK:0\\n");
    return 0;
"""
            elif v.value_type == "float":
                compare_code = f"""
    float read_val = read_bin("{filename}");
    if (read_val != 0.0f) {{
        printf("FAIL: read %f, expected 0.0\\n", read_val);
        return 1;
    }}
    printf("OK:0\\n");
    return 0;
"""
            elif v.value_type == "long":
                compare_code = f"""
    long read_val = read_bin("{filename}");
    if (read_val != 0) {{
        printf("FAIL: read %ld, expected 0\\n", read_val);
        return 1;
    }}
    printf("OK:0\\n");
    return 0;
"""
            else:  # int
                compare_code = f"""
    int read_val = read_bin("{filename}");
    if (read_val != 0) {{
        printf("FAIL: read %d, expected 0\\n", read_val);
        return 1;
    }}
    printf("OK:0\\n");
    return 0;
"""
            program = textwrap.dedent(f"""
            #include <stdio.h>
            #include <math.h>

            {self.solution}

            int main() {{
                {compare_code}
            }}
            """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if not self._check_solution_fclose():
            return "Ошибка: не найден вызов fclose() для закрытия файла", "требуется fclose"
        if not self._check_fopen_modes():
            return "Ошибка: функции должны открывать файл в бинарном режиме", "требуются бинарные режимы"

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

            if output.startswith(test.expected):
                return None
            return output, test.expected
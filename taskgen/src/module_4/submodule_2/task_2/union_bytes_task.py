from typing import Optional
import subprocess
import tempfile
import textwrap
from pathlib import Path

from src.base_module.base_task import BaseTaskClass, TestItem


_FIXED_VALUES = [0x12345678, 0xDEADBEEF, 0x00000000, 0xFFFFFFFF, 0xAABBCCDD]


def _generate_test_values(seed: int) -> list[int]:
    seed_val = 0x12345 ^ (seed * 0x9E3779B9 & 0xFFFFFFFF)
    return [
        0x12345678,
        seed_val & 0xFFFFFFFF,
        0xFF000000 | (seed & 0x00FFFFFF),
    ]


def _expected_output(value: int) -> str:
    b0 = (value >> 0)  & 0xFF
    b1 = (value >> 8)  & 0xFF
    b2 = (value >> 16) & 0xFF
    b3 = (value >> 24) & 0xFF
    return f"{b0:02x} {b1:02x} {b2:02x} {b3:02x}"


class UnionBytesTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.test_values = _generate_test_values(seed_value)

    def generate_task(self) -> str:
        return (
            "# Union: перекрытия памяти\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            "  Дано объявление:  \n\n"
            "  ```c\n"
            "  #include <stdint.h>\n\n"
            "  union ByteView {\n"
            "      uint32_t full;\n"
            "      unsigned char bytes;\n"
            "  };\n"
            "  ```\n\n"
            "  Напишите функцию `print_bytes`, которая:  \n"
            "  1. Принимает значение типа `uint32_t`.  \n"
            "  2. Записывает его в поле `full` объекта `union ByteView`.  \n"
            "  3. Выводит все 4 байта в памяти через пробел в формате `%02x`, начиная с `bytes[0]`.  \n\n"
            "  Сигнатура: `void print_bytes(uint32_t value)`  \n"
            "  Писать `main` не нужно — только тело функции.\n\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"print_bytes(0x{v:08X})",
                expected=_expected_output(v),
                compare_func=self._compare_default,
            )
            for v in self.test_values
        ]

    def _build_program_source(self, value: int) -> str:
        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <stdint.h>

            union ByteView {{
                uint32_t full;
                unsigned char bytes[4];
            }};

            {self.solution}

            int main(void) {{
                print_bytes(0x{value:08X}U);
                return 0;
            }}
        """)

    def _compile_and_run(self, value: int) -> tuple[bool, str]:
        program_source = self._build_program_source(value)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
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
        value = self.test_values[test_index]
        ok, result = self._compile_and_run(value)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
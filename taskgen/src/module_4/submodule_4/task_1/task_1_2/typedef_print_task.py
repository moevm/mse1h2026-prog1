from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    base_type: str
    alias: str
    fmt: str 
    precision: int 


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(base_type="float",  alias="f32", fmt="%.2f", precision=2),
    1: VariantSpec(base_type="double", alias="f64", fmt="%.4f", precision=4),
    2: VariantSpec(base_type="float",  alias="f32", fmt="%.3f", precision=3),
    3: VariantSpec(base_type="double", alias="f64", fmt="%.6f", precision=6),
}

_TEST_VALUES = [
    3.14159265,
    0.0,
    -1.5,
    100.0,
    0.001,
    42.123456789,
    1.00000001, 
    1.23456789012345, 
    0.00012345678,
    123456.78901234,
]


class TypedefPrintTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        example_val = 3.14159265
        expected_ex = f"{example_val:{'.'+str(v.precision)+'f'}}"
        return (
            "# Typedef: псевдоним и формат вывода вещественного типа\n\n"
            "### Задание №1.2\n\n"
            "- **Формулировка:**  \n"
            f"  Объявите псевдоним `{v.alias}` для типа `{v.base_type}`.  \n"
            f"  Напишите функцию `print_val`, которая принимает значение типа `{v.alias}`  \n"
            f"  и выводит его в формате `{v.fmt}` (без символа новой строки в конце не нужно — добавьте `\\n`).  \n\n"
            f"  Сигнатура: `void print_val({v.alias} x)`  \n\n"
            "  Писать `main` не нужно — только псевдоним и тело функции.  \n\n"
            "- **Пример:**  \n"
            f"  Вызов: `print_val({example_val})`  \n"
            f"  Вывод: `{expected_ex}`"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"print_val({val})",
                expected=f"{val:{'.'+str(v.precision)+'f'}}",
                compare_func=self._compare_default,
            )
            for val in _TEST_VALUES
        ]

    def _build_program_source(self, val: float) -> str:
        v = self.variant
        if v.base_type == "float":
            val_literal = f"{val}f"
        else:
            val_literal = str(val)

        return textwrap.dedent(f"""\
            #include <stdio.h>

            {self.solution}

            int main(void) {{
                print_val(({v.alias}){val_literal});
                return 0;
            }}
        """)

    def _compile_and_run(self, val: float) -> tuple[bool, str]:
        source = self._build_program_source(val)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(source, encoding="utf-8")
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
        
    def _check_typedef_size(self) -> tuple[bool, str]:
        v = self.variant
        source = textwrap.dedent(f"""\
            #include <stdio.h>
            #include <assert.h>

            {self.solution}

            int main(void) {{
                if (sizeof({v.alias}) != sizeof({v.base_type})) {{
                    printf("size_mismatch: got %zu expected %zu\\n",
                        sizeof({v.alias}), sizeof({v.base_type}));
                    return 1;
                }}
                printf("ok\\n");
                return 0;
            }}
        """)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_size.c"
            exe_path = tmp_path / "check_size.x"
            src_path.write_text(source, encoding="utf-8")
            cp = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if cp.returncode != 0:
                return False, cp.stdout.decode()
            rp = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            out = rp.stdout.decode().strip()
            return out == "ok", out

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self.tests.index(test) == 0:
            size_ok, size_msg = self._check_typedef_size()
            if not size_ok:
                return size_msg, f"sizeof({self.variant.alias}) == sizeof({self.variant.base_type})"

        test_index = self.tests.index(test)
        val = _TEST_VALUES[test_index]
        ok, result = self._compile_and_run(val)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
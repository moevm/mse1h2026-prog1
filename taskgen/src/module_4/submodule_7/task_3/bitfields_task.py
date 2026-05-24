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
    struct_def: str
    pack_sig: str
    unpack_sig: str
    fields: list[str]
    format_str: str
    test_cases: list


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        struct_def="typedef struct { unsigned int r : 5; unsigned int g : 5; unsigned int b : 5; unsigned int a : 1; } Color16;",
        pack_sig="Color16 pack(unsigned r, unsigned g, unsigned b, unsigned a)",
        unpack_sig="void unpack(Color16 c)",
        fields=["r", "g", "b", "a"],
        format_str="%u %u %u %u",
        test_cases=[
            ((10, 20, 30, 1), "10 20 30 1"),
            ((0, 0, 0, 0), "0 0 0 0"),
            ((31, 31, 31, 1), "31 31 31 1"),
        ],
    ),
    1: VariantSpec(
        struct_def="typedef struct { unsigned int day : 5; unsigned int month : 4; unsigned int year : 7; } Date;",
        pack_sig="Date pack(unsigned day, unsigned month, unsigned year)",
        unpack_sig="void unpack(Date d)",
        fields=["day", "month", "year"],
        format_str="%u %u %u",
        test_cases=[
            ((15, 7, 100), "15 7 100"),
            ((1, 1, 0), "1 1 0"),
            ((31, 12, 127), "31 12 127"),
        ],
    ),
    2: VariantSpec(
        struct_def="typedef struct { unsigned int hour : 5; unsigned int min : 6; unsigned int sec : 6; } Time;",
        pack_sig="Time pack(unsigned hour, unsigned min, unsigned sec)",
        unpack_sig="void unpack(Time t)",
        fields=["hour", "min", "sec"],
        format_str="%u %u %u",
        test_cases=[
            ((23, 59, 59), "23 59 59"),
            ((0, 0, 0), "0 0 0"),
            ((12, 30, 45), "12 30 45"),
        ],
    ),
    3: VariantSpec(
        struct_def="typedef struct { unsigned int x : 6; unsigned int y : 6; unsigned int flags : 4; } Tile;",
        pack_sig="Tile pack(unsigned x, unsigned y, unsigned flags)",
        unpack_sig="void unpack(Tile t)",
        fields=["x", "y", "flags"],
        format_str="%u %u %u",
        test_cases=[
            ((63, 63, 15), "63 63 15"),
            ((0, 0, 0), "0 0 0"),
            ((10, 20, 5), "10 20 5"),
        ],
    ),
}


class BitFieldsTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.test_extra = []

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Блок для самых умных: битовые поля\n\n"
            "### Задание №3\n\n"
            "- **Формулировка:**  \n"
            "  Вам дано объявление структуры с битовыми полями (соответствует вашему варианту).  \n"
            "  Напишите функцию `pack` и функцию `unpack`:\n"
            "  - `pack` принимает отдельные значения полей и возвращает заполненную структуру.\n"
            "  - `unpack` принимает структуру и выводит значения всех её полей в стандартный вывод в порядке, указанном в таблице.\n\n"
            "  Писать `main()` не нужно.\n\n"
            f"**Объявление структуры:**\n"
            f"```c\n{v.struct_def}\n```\n\n"
            f"**Сигнатура `pack`:** `{v.pack_sig}`\n"
            f"**Сигнатура `unpack`:** `{v.unpack_sig}`\n\n"
            f"Порядок вывода в `unpack`: `{' '.join(v.fields)}`\n"
            f"Формат вывода: `{v.format_str}`\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _extract_struct_name(self, struct_def: str) -> str:
        match = re.search(r'}\s*(\w+)\s*;', struct_def)
        return match.group(1) if match else ""

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        for i, (input_vals, expected_str) in enumerate(v.test_cases):
            test = TestItem(
                input_str=f"inputs = {input_vals}",
                showed_input=f"pack(...) → unpack()",
                expected=expected_str,
                compare_func=self._compare_default,
            )
            self.tests.append(test)
            self.test_extra.append({
                "input_vals": input_vals,
                "expected": expected_str,
            })

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        input_vals = extra["input_vals"]
        pack_args = ", ".join(str(x) for x in input_vals)
        struct_name = self._extract_struct_name(v.struct_def)
        main_code = f"""
    {struct_name} result = pack({pack_args});
    unpack(result);
    return 0;
"""
        program = textwrap.dedent(f"""
        #include <stdio.h>
        #include <stdlib.h>

        {v.struct_def}

        {self.solution}

        int main() {{
            {main_code}
        }}
        """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
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

            if output == test.expected:
                return None
            return output, test.expected
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    union_name: str
    int_field: str
    float_field: str
    float_type: str
    int_tag: str
    float_tag: str
    signature: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        union_name="Value",
        int_field="count",
        float_field="measurement",
        float_type="double",
        int_tag="i",
        float_tag="d",
        signature="union Value fill_Value(char tag, double value)",
    ),
    1: VariantSpec(
        union_name="Payload",
        int_field="code",
        float_field="ratio",
        float_type="float",
        int_tag="i",
        float_tag="f",
        signature="union Payload fill_Payload(char tag, double value)",
    ),
    2: VariantSpec(
        union_name="Reading",
        int_field="raw",
        float_field="voltage",
        float_type="float",
        int_tag="n",
        float_tag="v",
        signature="union Reading fill_Reading(char tag, double value)",
    ),
    3: VariantSpec(
        union_name="Data",
        int_field="flags",
        float_field="rate",
        float_type="double",
        int_tag="i",
        float_tag="d",
        signature="union Data fill_Data(char tag, double value)",
    ),
}


def _generate_values(seed: int) -> tuple[int, float]:
    int_val = 10 + (seed % 90)
    flt_val = (seed % 100) / 10.0 + 1.0
    return int_val, flt_val


class UnionFillTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.int_val, self.flt_val = _generate_values(seed_value)

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Union: использование\n\n"
            "### Задание №1\n\n"
            "- **Формулировка:**  \n"
            f"  Дано объявление `union`:  \n\n"
            "  ```c\n"
            f"  union {v.union_name} {{\n"
            f"      int {v.int_field};\n"
            f"      {v.float_type} {v.float_field};\n"
            "  };\n"
            "  ```\n\n"
            f"  Напишите функцию со следующей сигнатурой:  \n"
            f"  `{v.signature}`  \n\n"
            "  Функция принимает тег (`char tag`) и числовое значение (`double value`).  \n"
            f"  Если `tag == '{v.int_tag}'` — записывает `(int)value` в поле `{v.int_field}`.  \n"
            f"  Если `tag == '{v.float_tag}'` — записывает `({v.float_type})value` в поле `{v.float_field}`.  \n"
            "  Возвращает заполненный union.  \n\n"
            "  Писать `main` не нужно — только тело функции.\n\n"
            f"- **Пример 1:** `fill_{v.union_name}('{v.int_tag}', {float(self.int_val)})` "
            f"→ `{v.int_field} == {self.int_val}`  \n"
            f"- **Пример 2:** `fill_{v.union_name}('{v.float_tag}', {self.flt_val})` "
            f"→ `{v.float_field} ≈ {self.flt_val:.2f}`"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"fill_{v.union_name}('{v.int_tag}', {float(self.int_val)}) → {v.int_field}",
                expected=str(self.int_val),
                compare_func=self._compare_default,
            ),
            TestItem(
                input_str="",
                showed_input=f"fill_{v.union_name}('{v.float_tag}', {self.flt_val}) → {v.float_field}",
                expected=f"{self.flt_val:.2f}",
                compare_func=self._compare_default,
            ),
        ]

    def _build_program_source(self, test_index: int) -> str:
        v = self.variant

        union_decl = (
            f"union {v.union_name} {{\n"
            f"    int {v.int_field};\n"
            f"    {v.float_type} {v.float_field};\n"
            f"}};"
        )

        if test_index == 0:
            call = f"fill_{v.union_name}('{v.int_tag}', {float(self.int_val)})"
            check_and_print = f'printf("%d\\n", result.{v.int_field});'
        else:
            call = f"fill_{v.union_name}('{v.float_tag}', {self.flt_val})"
            check_and_print = f'printf("%.2f\\n", (double)result.{v.float_field});'

        return textwrap.dedent(f"""\
            #include <stdio.h>

            {union_decl}

            {self.solution}

            int main(void) {{
                union {v.union_name} result = {call};
                {check_and_print}
                return 0;
            }}
        """)

    def _compile_and_run(self, test_index: int) -> tuple[bool, str]:
        program_source = self._build_program_source(test_index)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            compile_proc = subprocess.run(
                [
                    "gcc", "-std=c11", "-O2",
                    "-Werror=float-conversion",  
                    str(src_path), "-o", str(exe_path),
                ],
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
        ok, result = self._compile_and_run(test_index)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
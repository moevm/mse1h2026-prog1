from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    type_name: str
    offset: int


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(type_name="short",  offset=2),
    1: VariantSpec(type_name="int",    offset=4),
    2: VariantSpec(type_name="float",  offset=4),
    3: VariantSpec(type_name="double", offset=8),
}


class StructAlignTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        return (
            "# Struct: выравнивание (alignment)\n\n"
            "### Задание №4\n\n"
            "- **Формулировка:**  \n"
            "  Дана структура (платформа x86-64, компилятор gcc):  \n\n"
            "  ```c\n"
            "  struct S {\n"
            "      char a;\n"
            f"      {self.variant.type_name} b;\n"
            "      char c;\n"
            "  };\n"
            "  ```\n\n"
            "  По какому смещению (в байтах от начала структуры) расположено поле `b`?  \n"
            "  Введите целое число (смещение)."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"TYPE = {self.variant.type_name}",
                expected=str(self.variant.offset),
                compare_func=self._compare_default,
            )
        ]

    def _build_verification_source(self) -> str:
        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <stddef.h>

            struct S {{
                char a;
                {self.variant.type_name} b;
                char c;
            }};

            int main(void) {{
                printf("%zu\\n", offsetof(struct S, b));
                return 0;
            }}
        """)

    def _get_real_offset(self) -> Optional[int]:
        source = self._build_verification_source()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "verify.c"
            exe_path = tmp_path / "verify.x"

            src_path.write_text(source, encoding="utf-8")
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if compile_proc.returncode != 0:
                return None
            run_proc = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            if run_proc.returncode != 0:
                return None
            try:
                return int(run_proc.stdout.decode().strip())
            except ValueError:
                return None

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        student_answer = self.solution.strip()

        real_offset = self._get_real_offset()
        expected = str(real_offset) if real_offset is not None else test.expected

        if student_answer == expected:
            return None
        return f"Неверно. Ваш ответ: {student_answer}", "[скрыто]"
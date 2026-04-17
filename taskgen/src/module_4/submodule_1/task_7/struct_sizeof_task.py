from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    fields_decl: str
    fields_display: str
    sizeof_val: int 


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        fields_decl="char a;\n    int b;",
        fields_display="char a; int b;",
        sizeof_val=8,
    ),
    1: VariantSpec(
        fields_decl="char a;\n    short b;\n    double c;",
        fields_display="char a; short b; double c;",
        sizeof_val=16,
    ),
    2: VariantSpec(
        fields_decl="int a;\n    char b;\n    int c;",
        fields_display="int a; char b; int c;",
        sizeof_val=12,
    ),
    3: VariantSpec(
        fields_decl="char a;\n    char b;\n    int c;\n    double d;",
        fields_display="char a; char b; int c; double d;",
        sizeof_val=16,
    ),
    4: VariantSpec(
        fields_decl="short a;\n    char b;\n    double c;\n    int d;",
        fields_display="short a; char b; double c; int d;",
        sizeof_val=24,
    ),
}


class StructSizeofTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        fields_lines = "\n".join(
            f"      {f.strip()};"
            for f in self.variant.fields_display.split(";")
            if f.strip()
        )
        return (
            "# Struct: sizeof(struct)\n\n"
            "### Задание №7\n\n"
            "- **Формулировка:**  \n"
            "  Дана структура (платформа x86-64, компилятор gcc):  \n\n"
            "  ```c\n"
            "  struct S {\n"
            f"{fields_lines}\n"
            "  };\n"
            "  ```\n\n"
            "  Чему равен `sizeof(struct S)`?  \n"
            "  Введите целое число."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"поля: {self.variant.fields_display}",
                expected=str(self.variant.sizeof_val),
                compare_func=self._compare_default,
            )
        ]

    def _build_verification_source(self) -> str:
        return textwrap.dedent(f"""\
            #include <stdio.h>

            struct S {{
                {self.variant.fields_decl}
            }};

            int main(void) {{
                printf("%zu\\n", sizeof(struct S));
                return 0;
            }}
        """)

    def _get_real_sizeof(self) -> Optional[int]:
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

        real_sizeof = self._get_real_sizeof()
        expected = str(real_sizeof) if real_sizeof is not None else test.expected

        if student_answer == expected:
            return None
        return f"Неверно. Ваш ответ: {student_answer}", "[скрыто]"
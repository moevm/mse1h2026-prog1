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
    field_names: tuple[str, ...] 
    field_types: tuple[str, ...]
    padding: int


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        fields_decl="char a;\n    int b;",
        fields_display="char a; int b;",
        field_names=("a", "b"),
        field_types=("char", "int"),
        padding=3,
    ),
    1: VariantSpec(
        fields_decl="char a;\n    short b;\n    double c;",
        fields_display="char a; short b; double c;",
        field_names=("a", "b", "c"),
        field_types=("char", "short", "double"),
        padding=5,
    ),
    2: VariantSpec(
        fields_decl="int a;\n    char b;\n    int c;",
        fields_display="int a; char b; int c;",
        field_names=("a", "b", "c"),
        field_types=("int", "char", "int"),
        padding=3,
    ),
    3: VariantSpec(
        fields_decl="char a;\n    char b;\n    int c;\n    double d;",
        fields_display="char a; char b; int c; double d;",
        field_names=("a", "b", "c", "d"),
        field_types=("char", "char", "int", "double"),
        padding=2,
    ),
    4: VariantSpec(
        fields_decl="short a;\n    char b;\n    double c;\n    int d;",
        fields_display="short a; char b; double c; int d;",
        field_names=("a", "b", "c", "d"),
        field_types=("short", "char", "double", "int"),
        padding=9,
    ),
}

_TYPE_SIZES: dict[str, int] = {
    "char": 1,
    "short": 2,
    "int": 4,
    "float": 4,
    "double": 8,
    "long": 8,
}


class StructPaddingTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        fields_lines = "\n".join(
            f"      {f};" if not f.strip().endswith(";") else f"      {f}"
            for f in self.variant.fields_display.split(";")
            if f.strip()
        )
        return (
            "# Struct: выравнивание (padding)\n\n"
            "### Задание №5\n\n"
            "- **Формулировка:**  \n"
            "  Дана структура (платформа x86-64, компилятор gcc):  \n\n"
            "  ```c\n"
            "  struct S {\n"
            f"{fields_lines}\n"
            "  };\n"
            "  ```\n\n"
            "  Сколько байт padding (выравнивающих пустот) добавляет компилятор "
            "в эту структуру?  \n"
            "  Введите целое число."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"поля: {self.variant.fields_display}",
                expected=str(self.variant.padding),
                compare_func=self._compare_default,
            )
        ]

    def _build_verification_source(self) -> str:
        """
        Вычисляет padding как:
            sizeof(struct S) - сумма sizeof каждого поля
        """
        field_names = self.variant.field_names
        sizeof_sum = " + ".join(
            f"sizeof(((struct S*)0)->{name})"
            for name in field_names
        )
        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <stddef.h>

            struct S {{
                {self.variant.fields_decl}
            }};

            int main(void) {{
                size_t total = sizeof(struct S);
                size_t data  = {sizeof_sum};
                printf("%zu\\n", total - data);
                return 0;
            }}
        """)

    def _get_real_padding(self) -> Optional[int]:
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

        real_padding = self._get_real_padding()
        expected = str(real_padding) if real_padding is not None else test.expected

        if student_answer == expected:
            return None
        return f"Неверно. Ваш ответ: {student_answer}", "[скрыто]"
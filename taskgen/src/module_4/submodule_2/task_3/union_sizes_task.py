from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    fields_display: str  
    fields_decl: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        fields_display="char a; int b; double c;",
        fields_decl="char a;\n    int b;\n    double c;",
    ),
    1: VariantSpec(
        fields_display="int a; float b; char c[4];",
        fields_decl="int a;\n    float b;\n    char c[4];",
    ),
    2: VariantSpec(
        fields_display="char a; char b; short c;",
        fields_decl="char a;\n    char b;\n    short c;",
    ),
    3: VariantSpec(
        fields_display="short a; long b; float c;",
        fields_decl="short a;\n    long b;\n    float c;",
    ),
    4: VariantSpec(
        fields_display="int a; int b; char c[8];",
        fields_decl="int a;\n    int b;\n    char c[8];",
    ),
}


def _build_reference_source(fields_decl: str) -> str:
    return textwrap.dedent(f"""\
        #include <stdio.h>
        #include <stddef.h>

        struct S {{
            {fields_decl}
        }};

        union U {{
            {fields_decl}
        }};

        int main(void) {{
            printf("%zu %zu\\n", sizeof(struct S), sizeof(union U));
            return 0;
        }}
    """)


def _get_reference_sizes(fields_decl: str) -> Optional[tuple[int, int]]:
    source = _build_reference_source(fields_decl)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_path = tmp_path / "ref.c"
        exe_path = tmp_path / "ref.x"

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
            parts = run_proc.stdout.decode().strip().split()
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return None


class UnionSizesTask(BaseTaskClass):
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
            "# Union: отличие от struct\n\n"
            "### Задание №3\n\n"
            "- **Формулировка:**  \n"
            "  Объявите структуру `S` и объединение `U` с одинаковым набором полей:  \n\n"
            "  ```c\n"
            f"{fields_lines}\n"
            "  ```\n\n"
            "  Напишите функцию `get_sizes`, которая возвращает через выходные параметры "
            "размеры `struct S` и `union U`:  \n\n"
            "  `void get_sizes(size_t *s_size, size_t *u_size)`  \n\n"
            "  Писать `main` не нужно — только объявления `struct S`, `union U` и тело функции."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        ref = _get_reference_sizes(self.variant.fields_decl)
        if ref is not None:
            s_size, u_size = ref
        else:
            s_size, u_size = 0, 0

        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"поля: {self.variant.fields_display}",
                expected=f"{s_size} {u_size}",
                compare_func=self._compare_default,
            )
        ]

    def _build_program_source(self) -> str:
        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <stddef.h>

            {self.solution}

            int main(void) {{
                size_t s_size, u_size;
                get_sizes(&s_size, &u_size);
                printf("%zu %zu\\n", s_size, u_size);
                return 0;
            }}
        """)

    def _compile_and_run(self) -> tuple[bool, str]:
        program_source = self._build_program_source()

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
        ref = _get_reference_sizes(self.variant.fields_decl)
        expected = f"{ref[0]} {ref[1]}" if ref is not None else test.expected

        ok, result = self._compile_and_run()
        if ok:
            if self._compare_default(result, expected):
                return None
            return result, expected
        return result, expected
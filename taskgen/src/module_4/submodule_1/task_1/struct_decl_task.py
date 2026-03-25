from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem

@dataclass(frozen=True)
class FieldSpec:
    field_name: str
    c_type: str
    comment: str = ""
    c_type_size: Optional[int] = None
    type_size_bytes: Optional[int] = None

@dataclass(frozen=True)
class VariantSpec:
    name: str
    entity: str
    fields: tuple[FieldSpec, ...]
    setup_lines: tuple[str, ...]
    expected_output: str

# Варианты заданий
_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        name="Student",
        entity="студента университета",
        fields=(
            FieldSpec("name", "char[32]", "имя", c_type_size=32),
            FieldSpec("age", "int", "возраст", type_size_bytes=4),
            FieldSpec("gpa", "float", "средний балл", type_size_bytes=4),
        ),
        setup_lines=(
            'strncpy(obj.name, "Alice", sizeof(obj.name) - 1);',
            'obj.name[sizeof(obj.name) - 1] = \'\\0\';',
            "obj.age = 20;",
            "obj.gpa = 4.5f;",
            'printf("%s %d %.1f\\n", obj.name, obj.age, obj.gpa);',
        ),
        expected_output="Alice 20 4.5",
    ),
    1: VariantSpec(
        name="Book",
        entity="книги в библиотеке",
        fields=(
            FieldSpec("title", "char[64]", "название книги", c_type_size=64),
            FieldSpec("pages", "int", "количество страниц", type_size_bytes=4),
            FieldSpec("price", "double", "цена", type_size_bytes=8),
        ),
        setup_lines=(
            'strncpy(obj.title, "C_Primer", sizeof(obj.title) - 1);',
            'obj.title[sizeof(obj.title) - 1] = \'\\0\';',
            "obj.pages = 896;",
            "obj.price = 1999.50;",
            'printf("%s %d %.2f\\n", obj.title, obj.pages, obj.price);',
        ),
        expected_output="C_Primer 896 1999.50",
    ),
    2: VariantSpec(
        name="Point2D",
        entity="точки на плоскости",
        fields=(
            FieldSpec("x", "double", "координата x", type_size_bytes=8),
            FieldSpec("y", "double", "координата y", type_size_bytes=8),
            FieldSpec("label", "int", "метка точки", type_size_bytes=4),
        ),
        setup_lines=(
            "obj.x = 1.25;",
            "obj.y = -3.50;",
            "obj.label = 7;",
            'printf("%.2f %.2f %d\\n", obj.x, obj.y, obj.label);',
        ),
        expected_output="1.25 -3.50 7",
    ),
    3: VariantSpec(
        name="Rectangle",
        entity="прямоугольник",
        fields=(
            FieldSpec("width", "double", "ширина", type_size_bytes=8),
            FieldSpec("height", "double", "высота", type_size_bytes=8),
            FieldSpec("color", "char[16]", "цвет", c_type_size=16),
        ),
        setup_lines=(
            "obj.width = 5.5;",
            "obj.height = 3.0;",
            'strncpy(obj.color, "red", sizeof(obj.color) - 1);',
            'obj.color[sizeof(obj.color) - 1] = \'\\0\';',
            'printf("%.1f %.1f %s\\n", obj.width, obj.height, obj.color);',
        ),
        expected_output="5.5 3.0 red",
    ),
    4: VariantSpec(
        name="Employee",
        entity="сотрудника компании",
        fields=(
            FieldSpec("name", "char[32]", "имя сотрудника", c_type_size=32),
            FieldSpec("id", "int", "табельный номер", type_size_bytes=4),
            FieldSpec("salary", "double", "зарплата", type_size_bytes=8),
        ),
        setup_lines=(
            'strncpy(obj.name, "Bob", sizeof(obj.name) - 1);',
            'obj.name[sizeof(obj.name) - 1] = \'\\0\';',
            "obj.id = 42;",
            "obj.salary = 12345.75;",
            'printf("%s %d %.2f\\n", obj.name, obj.id, obj.salary);',
        ),
        expected_output="Bob 42 12345.75",
    ),
}

class StructDeclTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        fields_md = "\n".join(
            f"      - `{field.field_name}` — тип `{field.c_type}`"
            + (f" ({field.comment})" if field.comment else "")
            for field in self.variant.fields
        )

        task_text = (
            "# Struct: объявление структуры\n\n"
            "### Задание №1\n\n"
            "- **Формулировка:**  \n"
            f"  Объявите структуру `{self.variant.name}`, описывающую {self.variant.entity}.  \n"
            "  Структура должна содержать следующие поля:\n\n"
            f"{fields_md}\n\n"
            "  Напишите **только объявление структуры**.  \n"
            "  Писать `main()` не нужно.  \n"
            "  Объявление может быть выполнено с typedef или без него."
        )
        return task_text

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"seed % {len(_VARIANTS)} = {self.variant_index} → {self.variant.name}",
                expected=self.variant.expected_output,
                compare_func=self._compare_default,
            )
        ]

    def _build_program_source(self, type_expr: str) -> str:
        # Сбор assert'ов для проверки типов и размеров
        checks = []
        for f in self.variant.fields:
            if f.c_type.startswith("char[") and f.c_type_size:
                checks.append(f"assert(sizeof(obj.{f.field_name}) == {f.c_type_size});")
            elif f.c_type in ("int", "float", "double") and f.type_size_bytes:
                checks.append(f"assert(sizeof(obj.{f.field_name}) == {f.type_size_bytes});")

        setup_block = textwrap.indent("\n".join(list(self.variant.setup_lines) + checks), "    ")

        return textwrap.dedent(
            f"""
            #include <stdio.h>
            #include <string.h>
            #include <assert.h>

            {self.solution}

            int main(void) {{
                {type_expr} obj;
            {setup_block}
                return 0;
            }}
            """
        ).strip() + "\n"

    def _compile_and_run(self, type_expr: str) -> tuple[bool, str]:
        program_source = self._build_program_source(type_expr)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")

            compile_cmd = ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)]
            compile_proc = subprocess.run(
                compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run([str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            output = "\n".join(part for part in (run_proc.stdout.decode().strip(), run_proc.stderr.decode().strip()) if part)
            if run_proc.returncode != 0:
                return False, output
            return True, output

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        attempts = [self.variant.name, f"struct {self.variant.name}"]
        errors: list[str] = []
        for type_expr in attempts:
            ok, result = self._compile_and_run(type_expr)
            if ok:
                if self._compare_default(result, test.expected):
                    return None
                return result, test.expected
            errors.append(result)
        return "\n\n".join(errors), test.expected
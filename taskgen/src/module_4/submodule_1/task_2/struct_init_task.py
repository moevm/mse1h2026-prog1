from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Paula",
    "Quentin", "Rose", "Steve", "Tina", "Ulysses", "Vera", "Will", "Xena",
    "Yves", "Zoe",
]


@dataclass(frozen=True)
class VariantSpec:
    type_name: str
    typedef_decl: str
    signature: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        type_name="Student",
        typedef_decl="typedef struct { char name[32]; int age; float gpa; } Student;",
        signature="Student init_Student(const char *name, int age, float gpa)",
    ),
    1: VariantSpec(
        type_name="Book",
        typedef_decl="typedef struct { char title[64]; int pages; double price; } Book;",
        signature="Book init_Book(const char *title, int pages, double price)",
    ),
    2: VariantSpec(
        type_name="Point2D",
        typedef_decl="typedef struct { double x; double y; int label; } Point2D;",
        signature="Point2D init_Point2D(double x, double y, int label)",
    ),
    3: VariantSpec(
        type_name="Rectangle",
        typedef_decl="typedef struct { double width; double height; char color[16]; } Rectangle;",
        signature="Rectangle init_Rectangle(double width, double height, const char *color)",
    ),
    4: VariantSpec(
        type_name="Employee",
        typedef_decl="typedef struct { char name[32]; int id; double salary; } Employee;",
        signature="Employee init_Employee(const char *name, int id, double salary)",
    ),
}


def _generate_values(seed: int, variant_index: int) -> dict:
    string_val = _NAMES[seed % len(_NAMES)]
    int_val = 10 + (seed % 90)
    float_val = (seed % 20) / 10.0 + 2.0
    double_val = (seed % 100) / 10.0 + 1.0
    double_val1 = (seed % 32) / 10.0 + 1.0
    double_val2 = (seed % 100) / 10.0 + 1.0

    if variant_index == 0:  # Student: name, age, gpa
        return {"name": string_val, "age": int_val, "gpa": float_val}
    elif variant_index == 1:  # Book: title, pages, price
        return {"title": string_val, "pages": int_val, "price": double_val}
    elif variant_index == 2:  # Point2D: x, y, label
        return {"x": double_val1, "y": double_val2, "label": int_val}
    elif variant_index == 3:  # Rectangle: width, height, color
        return {"width": double_val1, "height": double_val2, "color": string_val}
    elif variant_index == 4:  # Employee: name, id, salary
        return {"name": string_val, "id": int_val, "salary": double_val}
    return {}


def _call_expr(variant_index: int, vals: dict) -> str:
    if variant_index == 0:
        return f'init_Student("{vals["name"]}", {vals["age"]}, {vals["gpa"]}f)'
    elif variant_index == 1:
        return f'init_Book("{vals["title"]}", {vals["pages"]}, {vals["price"]})'
    elif variant_index == 2:
        return f'init_Point2D({vals["x"]}, {vals["y"]}, {vals["label"]})'
    elif variant_index == 3:
        return f'init_Rectangle({vals["width"]}, {vals["height"]}, "{vals["color"]}")'
    elif variant_index == 4:
        return f'init_Employee("{vals["name"]}", {vals["id"]}, {vals["salary"]})'
    return ""


def _assert_and_print_block(variant_index: int, vals: dict) -> str:
    if variant_index == 0:
        return textwrap.dedent(f"""\
            assert(strcmp(obj.name, "{vals["name"]}") == 0);
            assert(obj.age == {vals["age"]});
            assert(fabsf(obj.gpa - {vals["gpa"]}f) < 1e-5f);
            printf("%s %d %.1f\\n", obj.name, obj.age, obj.gpa);
        """)
    elif variant_index == 1:
        return textwrap.dedent(f"""\
            assert(strcmp(obj.title, "{vals["title"]}") == 0);
            assert(obj.pages == {vals["pages"]});
            assert(fabs(obj.price - {vals["price"]}) < 1e-9);
            printf("%s %d %.2f\\n", obj.title, obj.pages, obj.price);
        """)
    elif variant_index == 2:
        return textwrap.dedent(f"""\
            assert(fabs(obj.x - {vals["x"]}) < 1e-9);
            assert(fabs(obj.y - {vals["y"]}) < 1e-9);
            assert(obj.label == {vals["label"]});
            printf("%.1f %.1f %d\\n", obj.x, obj.y, obj.label);
        """)
    elif variant_index == 3:
        return textwrap.dedent(f"""\
            assert(fabs(obj.width - {vals["width"]}) < 1e-9);
            assert(fabs(obj.height - {vals["height"]}) < 1e-9);
            assert(strcmp(obj.color, "{vals["color"]}") == 0);
            printf("%.1f %.1f %s\\n", obj.width, obj.height, obj.color);
        """)
    elif variant_index == 4:
        return textwrap.dedent(f"""\
            assert(strcmp(obj.name, "{vals["name"]}") == 0);
            assert(obj.id == {vals["id"]});
            assert(fabs(obj.salary - {vals["salary"]}) < 1e-9);
            printf("%s %d %.2f\\n", obj.name, obj.id, obj.salary);
        """)
    return ""


def _expected_output(variant_index: int, vals: dict) -> str:
    if variant_index == 0:
        return f'{vals["name"]} {vals["age"]} {vals["gpa"]:.1f}'
    elif variant_index == 1:
        return f'{vals["title"]} {vals["pages"]} {vals["price"]:.2f}'
    elif variant_index == 2:
        return f'{vals["x"]:.1f} {vals["y"]:.1f} {vals["label"]}'
    elif variant_index == 3:
        return f'{vals["width"]:.1f} {vals["height"]:.1f} {vals["color"]}'
    elif variant_index == 4:
        return f'{vals["name"]} {vals["id"]} {vals["salary"]:.2f}'
    return ""


class StructInitTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.vals = _generate_values(seed_value, self.variant_index)

    def generate_task(self) -> str:
        return (
            "# Struct: инициализация переменной\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            f"  Дано объявление структуры:  \n"
            f"  `{self.variant.typedef_decl}`  \n\n"
            f"  Напишите функцию со следующей сигнатурой:  \n"
            f"  `{self.variant.signature}`  \n\n"
            "  Функция должна возвращать переменную указанного типа, инициализированную "
            "**одним выражением** внутри `return`, используя **designated initializer**.  \n\n"
            "  Писать `main`, `typedef` и вызов функции не нужно — только тело функции."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=_call_expr(self.variant_index, self.vals),
                expected=_expected_output(self.variant_index, self.vals),
                compare_func=self._compare_default,
            )
        ]

    def _build_program_source(self) -> str:
        variant = self.variant
        vals = self.vals
        call = _call_expr(self.variant_index, vals)
        checks = textwrap.indent(_assert_and_print_block(self.variant_index, vals), "    ")

        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <string.h>
            #include <math.h>
            #include <assert.h>

            {variant.typedef_decl}

            {self.solution}

            int main(void) {{
                {variant.type_name} obj = {call};
            {checks}
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

            compile_cmd = ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path), "-lm"]
            compile_proc = subprocess.run(
                compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            output = "\n".join(
                part for part in (run_proc.stdout.decode().strip(), run_proc.stderr.decode().strip()) if part
            )
            if run_proc.returncode != 0:
                return False, output
            return True, output

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        ok, result = self._compile_and_run()
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
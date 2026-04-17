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
    struct_decl: str  
    sort_field: str          
    sort_field_type: str 
    cmp_name: str  
    test_items: tuple[dict, ...]
    c_init_template: str
    c_print_template: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        type_name="Student",
        struct_decl="typedef struct { char name[32]; int age; float gpa; } Student;",
        sort_field="age",
        sort_field_type="int",
        cmp_name="cmp_Student",
        test_items=(
            {"name": "Alice", "age": 22, "gpa": 3.8},
            {"name": "Bob",   "age": 19, "gpa": 4.0},
            {"name": "Carol", "age": 25, "gpa": 3.2},
            {"name": "Dave",  "age": 20, "gpa": 3.5},
            {"name": "Eve",   "age": 21, "gpa": 3.9},
        ),
        c_init_template='{{"{name}", {age}, {gpa}f}}',
        c_print_template='printf("%d\\n", arr[i].age);',
    ),
    1: VariantSpec(
        type_name="Book",
        struct_decl="typedef struct { char title[64]; int pages; double price; } Book;",
        sort_field="pages",
        sort_field_type="int",
        cmp_name="cmp_Book",
        test_items=(
            {"title": "C_Primer",    "pages": 896,  "price": 1999.50},
            {"title": "SICP",        "pages": 657,  "price": 2500.00},
            {"title": "CLRS",        "pages": 1292, "price": 3200.00},
            {"title": "K&R",         "pages": 274,  "price": 800.00},
            {"title": "PragProg",    "pages": 352,  "price": 1500.00},
        ),
        c_init_template='{{"{title}", {pages}, {price}}}',
        c_print_template='printf("%d\\n", arr[i].pages);',
    ),
    2: VariantSpec(
        type_name="Point2D",
        struct_decl="typedef struct { double x; double y; int label; } Point2D;",
        sort_field="label",
        sort_field_type="int",
        cmp_name="cmp_Point2D",
        test_items=(
            {"x": 1.0, "y": 2.0,  "label": 10},
            {"x": 3.5, "y": 4.5,  "label": 20},
            {"x": 0.0, "y": 1.0,  "label": 15},
            {"x": 2.2, "y": 3.3,  "label": 5},
            {"x": 5.0, "y": 0.5,  "label": 30},
        ),
        c_init_template="{{{x}, {y}, {label}}}",
        c_print_template='printf("%d\\n", arr[i].label);',
    ),
    3: VariantSpec(
        type_name="Rectangle",
        struct_decl='typedef struct { double width; double height; char color[16]; } Rectangle;',
        sort_field="width",
        sort_field_type="double",
        cmp_name="cmp_Rectangle",
        test_items=(
            {"width": 5.5,  "height": 3.0, "color": "red"},
            {"width": 1.2,  "height": 7.0, "color": "blue"},
            {"width": 9.0,  "height": 2.5, "color": "green"},
            {"width": 3.3,  "height": 4.4, "color": "black"},
            {"width": 0.8,  "height": 1.1, "color": "white"},
        ),
        c_init_template='{{{width}, {height}, "{color}"}}',
        c_print_template='printf("%.1f\\n", arr[i].width);',
    ),
    4: VariantSpec(
        type_name="Employee",
        struct_decl="typedef struct { char name[32]; int id; double salary; } Employee;",
        sort_field="salary",
        sort_field_type="double",
        cmp_name="cmp_Employee",
        test_items=(
            {"name": "Alice", "id": 1, "salary": 75000.0},
            {"name": "Bob",   "id": 2, "salary": 92000.0},
            {"name": "Carol", "id": 3, "salary": 61000.0},
            {"name": "Dave",  "id": 4, "salary": 88000.0},
            {"name": "Eve",   "id": 5, "salary": 45000.0},
        ),
        c_init_template='{{"{name}", {id}, {salary}}}',
        c_print_template='printf("%.1f\\n", arr[i].salary);',
    ),
}


def _sorted_values(items: tuple[dict, ...], field: str, descending: bool) -> list:
    return sorted(
        [item[field] for item in items],
        reverse=descending,
    )


class QsortTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.descending = bool(seed_value % 2)

    def _dir_str(self) -> str:
        return "убыванию" if self.descending else "возрастанию"

    def generate_task(self) -> str:
        v = self.variant
        items_display = "\n".join(
            "  { " + ", ".join(f".{k}={val!r}" for k, val in item.items()) + " }"
            for item in v.test_items
        )
        expected = _sorted_values(v.test_items, v.sort_field, self.descending)
        expected_display = "\n".join(f"  {val}" for val in expected)
        return (
            "# stdlib: qsort\n\n"
            "### Задание №7\n\n"
            "- **Формулировка:**  \n"
            f"  Вам дано объявление структуры и заполненный массив объектов структуры:\n\n"
            f"  ```c\n"
            f"  {v.struct_decl}\n"
            f"  ```\n\n"
            f"  Массив из {len(v.test_items)} элементов передаётся в `sort_arr`.  \n\n"
            f"  Напишите:  \n"
            f"  1. Компаратор `{v.cmp_name}(const void*, const void*)`.  \n"
            f"  2. Функцию `void sort_arr({v.type_name} arr[], int n)`,  \n"
            f"     сортирующую по полю `{v.sort_field}` по **{self._dir_str()}**.  \n\n"
            "  Писать `main` не нужно.\n\n"
            "- **Пример входных данных:**\n"
            f"  ```\n{items_display}\n  ```\n\n"
            f"  Ожидаемые значения поля `{v.sort_field}` после сортировки:\n"
            f"  ```\n{expected_display}\n  ```"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        expected_vals = _sorted_values(v.test_items, v.sort_field, self.descending)
        expected_str = "\n".join(
            (f"{val:.1f}" if v.sort_field_type == "double" else str(val))
            for val in expected_vals
        )
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"sort_arr(arr, {len(v.test_items)}) по полю '{v.sort_field}' по {self._dir_str()}",
                expected=expected_str,
                compare_func=self._compare_default,
            )
        ]

    def _build_program_source(self) -> str:
        v = self.variant
        n = len(v.test_items)

        inits = ",\n        ".join(
            v.c_init_template.format(**item) for item in v.test_items
        )

        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <stdlib.h>
            #include <string.h>

            {v.struct_decl}

            {self.solution}

            int main(void) {{
                {v.type_name} arr[{n}] = {{
                    {inits}
                }};
                sort_arr(arr, {n});
                for (int i = 0; i < {n}; i++) {{
                    {v.c_print_template}
                }}
                return 0;
            }}
        """)

    def _compile_and_run(self) -> tuple[bool, str]:
        source = self._build_program_source()
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

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        ok, result = self._compile_and_run()
        if ok:
            result_norm = "\n".join(line.strip() for line in result.strip().splitlines())
            expected_norm = "\n".join(line.strip() for line in test.expected.strip().splitlines())
            if result_norm == expected_norm:
                return None
            return result_norm, expected_norm
        return result, test.expected
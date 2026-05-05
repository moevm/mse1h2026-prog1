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
    key_field: str
    key_type: str 
    cmp_name: str
    items: tuple[dict, ...]
    c_init_template: str   
    c_key_fmt: str   
    c_found_fmt: str 
    c_found_print: str  


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        type_name="Student",
        struct_decl="typedef struct { char name[32]; int age; float gpa; } Student;",
        key_field="age",
        key_type="int",
        cmp_name="cmp_Student",
        items=(
            {"name": "Bob",   "age": 19, "gpa": 4.0},
            {"name": "Dave",  "age": 20, "gpa": 3.5},
            {"name": "Eve",   "age": 21, "gpa": 3.9},
            {"name": "Alice", "age": 22, "gpa": 3.8},
            {"name": "Carol", "age": 25, "gpa": 3.2},
        ),
        c_init_template='{{"{name}", {age}, {gpa}f}}',
        c_key_fmt="%d",
        c_found_fmt="%.1f",
        c_found_print='printf("%.1f\\n", result->gpa);',
    ),
    1: VariantSpec(
        type_name="Book",
        struct_decl="typedef struct { char title[64]; int pages; double price; } Book;",
        key_field="pages",
        key_type="int",
        cmp_name="cmp_Book",
        items=(
            {"title": "K&R",      "pages": 274,  "price": 800.00},
            {"title": "PragProg", "pages": 352,  "price": 1500.00},
            {"title": "SICP",     "pages": 657,  "price": 2500.00},
            {"title": "C_Primer", "pages": 896,  "price": 1999.50},
            {"title": "CLRS",     "pages": 1292, "price": 3200.00},
        ),
        c_init_template='{{"{title}", {pages}, {price}}}',
        c_key_fmt="%d",
        c_found_fmt="%.2f",
        c_found_print='printf("%.2f\\n", result->price);',
    ),
    2: VariantSpec(
        type_name="Point2D",
        struct_decl="typedef struct { double x; double y; int label; } Point2D;",
        key_field="label",
        key_type="int",
        cmp_name="cmp_Point2D",
        items=(
            {"x": 1.0, "y": 2.0, "label": 10},
            {"x": 2.0, "y": 3.0, "label": 20},
            {"x": 3.0, "y": 4.0, "label": 30},
            {"x": 4.0, "y": 5.0, "label": 40},
            {"x": 5.0, "y": 6.0, "label": 50},
        ),
        c_init_template="{{{x}, {y}, {label}}}",
        c_key_fmt="%d",
        c_found_fmt="%.1f",
        c_found_print='printf("%.1f\\n", result->x);',
    ),
    3: VariantSpec(
        type_name="Rectangle",
        struct_decl='typedef struct { double width; double height; char color[16]; } Rectangle;',
        key_field="width",
        key_type="double",
        cmp_name="cmp_Rectangle",
        items=(
            {"width": 0.8, "height": 1.1, "color": "white"},
            {"width": 1.2, "height": 7.0, "color": "blue"},
            {"width": 3.3, "height": 4.4, "color": "black"},
            {"width": 5.5, "height": 3.0, "color": "red"},
            {"width": 9.0, "height": 2.5, "color": "green"},
        ),
        c_init_template='{{{width}, {height}, "{color}"}}',
        c_key_fmt="%.1f",
        c_found_fmt="%.1f",
        c_found_print='printf("%.1f\\n", result->height);',
    ),
    4: VariantSpec(
        type_name="Employee",
        struct_decl="typedef struct { char name[32]; int id; double salary; } Employee;",
        key_field="salary",
        key_type="double",
        cmp_name="cmp_Employee",
        items=(
            {"name": "Eve",   "id": 5, "salary": 45000.0},
            {"name": "Carol", "id": 3, "salary": 61000.0},
            {"name": "Alice", "id": 1, "salary": 75000.0},
            {"name": "Dave",  "id": 4, "salary": 88000.0},
            {"name": "Bob",   "id": 2, "salary": 92000.0},
        ),
        c_init_template='{{"{name}", {id}, {salary}}}',
        c_key_fmt="%.1f",
        c_found_fmt="%d",
        c_found_print='printf("%d\\n", result->id);',
    ),
}


class BsearchTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        items_display = "\n".join(
            "  { " + ", ".join(f".{k}={val!r}" for k, val in item.items()) + " }"
            for item in v.items
        )
        mid = v.items[len(v.items) // 2]
        key_val = mid[v.key_field]
        key_str = f"{key_val:.1f}" if v.key_type == "double" else str(key_val)

        if v.key_type == "int":
            key_type_sig = "int key"
        else:
            key_type_sig = "double key"

        return (
            "# stdlib: bsearch\n\n"
            "### Задание №8\n\n"
            "- **Формулировка:**  \n"
            f"  Структура:\n\n"
            f"  ```c\n  {v.struct_decl}\n  ```\n\n"
            f"  Вам дан массив структур указанного типа из {len(v.items)} элементов, отсортированных по полю "
            f"`{v.key_field}` по возрастанию.  \n\n"
            "  Напишите:  \n"
            f"  1. Компаратор `{v.cmp_name}(const void*, const void*)`.  \n"
            f"  2. Функцию `{v.type_name} *find_in_arr({v.type_name} arr[], int n, {key_type_sig})`,  \n"
            f"     которая возвращает указатель на элемент с `{v.key_field} == key`  \n"
            f"     или `NULL`, если не найден.  \n\n"
            "  Писать `main` не нужно.\n\n"
            "- **Пример данных:**\n"
            f"  ```\n{items_display}\n  ```\n\n"
            f"  `find_in_arr(arr, {len(v.items)}, {key_str})` → элемент найден  \n"
            f"  `find_in_arr(arr, {len(v.items)}, 999999)` → `NULL`"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        mid = v.items[len(v.items) // 2]
        key_val = mid[v.key_field]
        first = v.items[0]
        key_first = first[v.key_field]
        last = v.items[-1]
        key_last = last[v.key_field]
        missing_key = 999999 if v.key_type == "int" else 999999.0

        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"find_in_arr(arr, {len(v.items)}, {key_val}) → найти средний; "
                             f"find_in_arr(..., {key_first}) → первый; "
                             f"find_in_arr(..., {key_last}) → последний; "
                             f"find_in_arr(..., {missing_key}) → NULL",
                expected="found\nfound\nfound\nnull",
                compare_func=self._compare_default,
            )
        ]

    def _build_program_source(self) -> str:
        v = self.variant
        n = len(v.items)
        inits = ",\n        ".join(
            v.c_init_template.format(**item) for item in v.items
        )

        mid = v.items[n // 2]
        key_mid = mid[v.key_field]
        key_first = v.items[0][v.key_field]
        key_last = v.items[-1][v.key_field]

        if v.key_type == "int":
            missing = "999999"
            k_mid = str(key_mid)
            k_first = str(key_first)
            k_last = str(key_last)
        else:
            missing = "999999.0"
            k_mid = f"{key_mid}"
            k_first = f"{key_first}"
            k_last = f"{key_last}"

        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <stdlib.h>
            #include <string.h>

            {v.struct_decl}

            {self.solution}

            static void check(const char *label, {v.type_name} *result) {{
                if (result != NULL) {{
                    printf("found\\n");
                }} else {{
                    printf("null\\n");
                }}
            }}

            int main(void) {{
                {v.type_name} arr[{n}] = {{
                    {inits}
                }};

                check("mid",     find_in_arr(arr, {n}, {k_mid}));
                check("first",   find_in_arr(arr, {n}, {k_first}));
                check("last",    find_in_arr(arr, {n}, {k_last}));
                check("missing", find_in_arr(arr, {n}, {missing}));

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
            result_norm = "\n".join(l.strip() for l in result.strip().splitlines())
            expected_norm = "\n".join(l.strip() for l in test.expected.strip().splitlines())
            if result_norm == expected_norm:
                return None
            return result_norm, expected_norm
        return result, test.expected
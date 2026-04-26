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
    typedef_decl: str
    target_field: str
    field_type: str
    signature: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        type_name="Student",
        typedef_decl="typedef struct { char name[32]; int age; float gpa; } Student;",
        target_field="age",
        field_type="int",
        signature="int compute_Student(Student arr[], int n)",
    ),
    1: VariantSpec(
        type_name="Book",
        typedef_decl="typedef struct { char title[64]; int pages; double price; } Book;",
        target_field="pages",
        field_type="int",
        signature="int compute_Book(Book arr[], int n)",
    ),
    2: VariantSpec(
        type_name="Point2D",
        typedef_decl="typedef struct { double x; double y; int label; } Point2D;",
        target_field="label",
        field_type="int",
        signature="int compute_Point2D(Point2D arr[], int n)",
    ),
    3: VariantSpec(
        type_name="Rectangle",
        typedef_decl="typedef struct { double width; double height; char color[16]; } Rectangle;",
        target_field="width",
        field_type="double",
        signature="double compute_Rectangle(Rectangle arr[], int n)",
    ),
    4: VariantSpec(
        type_name="Employee",
        typedef_decl="typedef struct { char name[32]; int id; double salary; } Employee;",
        target_field="salary",
        field_type="double",
        signature="double compute_Employee(Employee arr[], int n)",
    ),
}

_TEST_ARRAYS: dict[int, list[dict]] = {
    0: [  # Student.age
        {"name": "Alice", "age": 20, "gpa": 3.5},
        {"name": "Bob",   "age": 22, "gpa": 4.0},
        {"name": "Carol", "age": 19, "gpa": 3.8},
    ],
    1: [  # Book.pages
        {"title": "C_Primer",    "pages": 300, "price": 29.99},
        {"title": "SICP",        "pages": 657, "price": 45.00},
        {"title": "Clean_Code",  "pages": 431, "price": 35.50},
    ],
    2: [  # Point2D.label
        {"x": 1.0, "y": 2.0, "label": 10},
        {"x": 3.5, "y": 4.5, "label": 20},
        {"x": 0.0, "y": 1.0, "label": 15},
    ],
    3: [  # Rectangle.width
        {"width": 3.0, "height": 2.0, "color": "red"},
        {"width": 5.5, "height": 1.5, "color": "blue"},
        {"width": 4.0, "height": 3.0, "color": "green"},
    ],
    4: [  # Employee.salary
        {"name": "Alice", "id": 1, "salary": 50000.0},
        {"name": "Bob",   "id": 2, "salary": 75000.0},
        {"name": "Carol", "id": 3, "salary": 60000.0},
    ],
}


def _compute_expected(variant_index: int, op: int) -> float:
    arr = _TEST_ARRAYS[variant_index]
    field = _VARIANTS[variant_index].target_field
    values = [row[field] for row in arr]
    if op == 0:
        return max(values)
    elif op == 1:
        return sum(values)
    else:
        return min(values)


def _format_expected(variant_index: int, value: float) -> str:
    if _VARIANTS[variant_index].field_type == "int":
        return str(int(value))
    else:
        return f"{value:.1f}"


def _array_init_lines(variant_index: int) -> str:
    arr = _TEST_ARRAYS[variant_index]
    type_name = _VARIANTS[variant_index].type_name
    lines = []
    for i, row in enumerate(arr):
        if variant_index == 0:
            lines.append(
                f'    strncpy(arr[{i}].name, "{row["name"]}", 31); '
                f'arr[{i}].age = {row["age"]}; arr[{i}].gpa = {row["gpa"]}f;'
            )
        elif variant_index == 1:
            lines.append(
                f'    strncpy(arr[{i}].title, "{row["title"]}", 63); '
                f'arr[{i}].pages = {row["pages"]}; arr[{i}].price = {row["price"]};'
            )
        elif variant_index == 2:
            lines.append(
                f'    arr[{i}].x = {row["x"]}; arr[{i}].y = {row["y"]}; arr[{i}].label = {row["label"]};'
            )
        elif variant_index == 3:
            lines.append(
                f'    arr[{i}].width = {row["width"]}; arr[{i}].height = {row["height"]}; '
                f'strncpy(arr[{i}].color, "{row["color"]}", 15);'
            )
        elif variant_index == 4:
            lines.append(
                f'    strncpy(arr[{i}].name, "{row["name"]}", 31); '
                f'arr[{i}].id = {row["id"]}; arr[{i}].salary = {row["salary"]};'
            )
    return "\n".join(lines)


def _printf_result(field_type: str) -> str:
    if field_type == "int":
        return 'printf("%d\\n", result);'
    else:
        return 'printf("%.1f\\n", result);'


class StructComputeTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.op = seed_value % 3
        self.variant = _VARIANTS[self.variant_index]
        self._expected_value = _compute_expected(self.variant_index, self.op)

    def generate_task(self) -> str:
        op_names = {0: "максимум", 1: "сумму", 2: "минимум"}
        arr = _TEST_ARRAYS[self.variant_index]
        field = self.variant.target_field

        arr_lines = "\n".join(
            "  { " + ", ".join(f".{k}={v!r}" for k, v in row.items()) + " }"
            for row in arr
        )

        return (
            "# Struct: доступ к полям\n\n"
            "### Задание №3\n\n"
            "- **Формулировка:**  \n"
            f"  Дано объявление структуры:  \n"
            f"  `{self.variant.typedef_decl}`  \n\n"
            f"  Напишите функцию со следующей сигнатурой:  \n"
            f"  `{self.variant.signature}`  \n\n"
            f"  Функция должна вычислить и вернуть **{op_names[self.op]}** "
            f"значений поля `{field}` всех элементов массива.  \n\n"
            "  Писать `main` не нужно — только тело функции.\n\n"
            f"- **Пример вызова:** `compute_{self.variant.type_name}(arr, 3)`,  \n"
            f"  где массив содержит:\n"
            f"  ```\n{arr_lines}\n  ```\n"
            f"  Ожидаемый результат: `{_format_expected(self.variant_index, self._expected_value)}`"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input=(
                    f"compute_{self.variant.type_name}(arr, 3), "
                    f"op={self.op} ({'max' if self.op==0 else 'sum' if self.op==1 else 'min'}), "
                    f"field={self.variant.target_field}"
                ),
                expected=_format_expected(self.variant_index, self._expected_value),
                compare_func=self._compare_default,
            )
        ]

    def _build_program_source(self) -> str:
        variant = self.variant
        init_lines = _array_init_lines(self.variant_index)
        printf_line = _printf_result(variant.field_type)

        return textwrap.dedent(f"""\
            #include <stdio.h>
            #include <string.h>

            {variant.typedef_decl}

            {self.solution}

            int main(void) {{
                {variant.type_name} arr[3];
            {init_lines}
                {variant.field_type} result = compute_{variant.type_name}(arr, 3);
                {printf_line}
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

            compile_cmd = ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)]
            compile_proc = subprocess.run(
                compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
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
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
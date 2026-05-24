from dataclasses import dataclass
from typing import Optional
import subprocess
import tempfile
import textwrap
import re
from pathlib import Path

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    outer_struct_def: str
    inner_struct_def: str
    find_best_sig: str
    inner_field: str
    format_spec: str
    type_name: str
    test_data: list


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        outer_struct_def="typedef struct { char name[32]; Coords location; int rating; } Place;",
        inner_struct_def="typedef struct { float lat; float lon; } Coords;",
        find_best_sig="Place *find_best(Place *arr, int n)",
        inner_field="location.lat",
        format_spec="%.2f",
        type_name="float",
        test_data=[
            ("Cafe\n12.34 56.78\n4", "12.34"),
            ("Library\n98.76 54.32\n5", "98.76"),
            ("Museum\n45.67 23.45\n3", "98.76"),
        ],
    ),
    1: VariantSpec(
        outer_struct_def="typedef struct { char title[64]; Date published; int sales; } Book;",
        inner_struct_def="typedef struct { int year; int month; } Date;",
        find_best_sig="Book *find_best(Book *arr, int n)",
        inner_field="published.year",
        format_spec="%d",
        type_name="int",
        test_data=[
            ("Book1\n2020 5\n100", "2020"),
            ("Book2\n2023 7\n200", "2023"),
            ("Book3\n2021 3\n150", "2023"),
        ],
    ),
    2: VariantSpec(
        outer_struct_def="typedef struct { Point center; double radius; char color[16]; } Circle;",
        inner_struct_def="typedef struct { double x; double y; } Point;",
        find_best_sig="Circle *find_best(Circle *arr, int n)",
        inner_field="center.x",
        format_spec="%.1f",
        type_name="double",
        test_data=[
            ("1.5 2.5\n3.0\nred", "1.5"),
            ("10.2 3.4\n5.0\ngreen", "10.2"),
            ("5.5 1.2\n4.0\nblue", "10.2"),
        ],
    ),
    3: VariantSpec(
        outer_struct_def="typedef struct { char event[32]; Time start; int duration; } Schedule;",
        inner_struct_def="typedef struct { int hours; int minutes; } Time;",
        find_best_sig="Schedule *find_best(Schedule *arr, int n)",
        inner_field="start.hours",
        format_spec="%d",
        type_name="int",
        test_data=[
            ("Meeting\n9 30\n60", "9"),
            ("Workshop\n14 0\n120", "14"),
            ("Break\n12 0\n15", "14"),
        ],
    ),
}


class NestedStructTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Блок для самых умных: структура в структуре\n\n"
            "### Задание №1\n\n"
            "- **Формулировка:**  \n"
            "  Вам даны объявления двух вложенных структур (соответствует вашему варианту).  \n"
            "  Напишите функцию `find_best`, которая:\n"
            "  1. Принимает массив структур и его размер.\n"
            "  2. Находит элемент массива, у которого значение поля вложенной структуры `inner_field` максимально.\n"
            "  3. Возвращает указатель на этот элемент.\n\n"
            "  Писать `main()` не нужно.\n\n"
            "**Объявления структур:**\n"
            f"```c\n{v.inner_struct_def}\n{v.outer_struct_def}\n```\n\n"
            f"**Сигнатура `find_best`:** `{v.find_best_sig}`\n\n"
            f"**Поле для сравнения:** `{v.inner_field}`\n\n"
            "Программа считывает из стандартного ввода `n = 3` объекта внешней структуры,\n"
            "вызывает `find_best` и выводит значение `inner_field` найденного элемента.\n\n"
            "Функция должна возвращать именно **указатель** на элемент массива, а не копию.\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _extract_type_name(self, def_str: str) -> str:
        """Извлекает имя типа из typedef определения (после последней '}')"""
        match = re.search(r'}\s*(\w+)\s*;', def_str)
        if match:
            return match.group(1)
        return def_str.rstrip(';').split()[-1]

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        input_data_lines = ["3"]
        for data_str, _ in v.test_data:
            input_data_lines.append(data_str)
        full_input = "\n".join(input_data_lines) + "\n"

        expected = v.test_data[1][1]

        test = TestItem(
            input_str=full_input.strip(),
            showed_input=f"3 объекта, ищем максимум {v.inner_field}",
            expected=f"OUT:{expected}",
            compare_func=self._compare_default,
        )
        self.tests.append(test)
        self.test_extra.append({
            "input_data": full_input,
            "expected_value": expected,
        })

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        outer_type = self._extract_type_name(v.outer_struct_def)

        if v.type_name == "int":
            print_code = f'printf("OUT:{v.format_spec}\\n", best->{v.inner_field});'
        elif v.type_name in ("float", "double"):
            print_code = f'printf("OUT:{v.format_spec}\\n", best->{v.inner_field});'
        else:
            print_code = f'printf("OUT:{v.format_spec}\\n", best->{v.inner_field});'

        if self.variant_index == 0:
            read_loop = """
    for (int i = 0; i < n; i++) {
        fscanf(stdin, "%s %f %f %d", arr[i].name, &arr[i].location.lat, &arr[i].location.lon, &arr[i].rating);
    }
"""
        elif self.variant_index == 1:
            read_loop = """
    for (int i = 0; i < n; i++) {
        fscanf(stdin, "%s %d %d %d", arr[i].title, &arr[i].published.year, &arr[i].published.month, &arr[i].sales);
    }
"""
        elif self.variant_index == 2:
            read_loop = """
    for (int i = 0; i < n; i++) {
        fscanf(stdin, "%lf %lf %lf %s", &arr[i].center.x, &arr[i].center.y, &arr[i].radius, arr[i].color);
    }
"""
        else:
            read_loop = """
    for (int i = 0; i < n; i++) {
        fscanf(stdin, "%s %d %d %d", arr[i].event, &arr[i].start.hours, &arr[i].start.minutes, &arr[i].duration);
    }
"""

        program = textwrap.dedent(f"""
        #include <stdio.h>
        #include <stdlib.h>

        {v.inner_struct_def}
        {v.outer_struct_def}

        {self.solution}

        int main() {{
            int n;
            scanf("%d", &n);
            {outer_type} *arr = malloc(n * sizeof(*arr));
            if (!arr) return 1;
            {read_loop}
            {outer_type} *best = find_best(arr, n);
            if (best) {{
                {print_code}
            }} else {{
                printf("OUT:error\\n");
            }}
            free(arr);
            return 0;
        }}
        """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        try:
            idx = self.tests.index(test)
        except ValueError:
            return "Test not found", "unknown"
        extra = self.test_extra[idx]

        program_source = self._build_test_program(extra)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "test_program.c"
            exe_path = tmp_path / "test_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                cwd=tmpdir,
            )
            if compile_proc.returncode != 0:
                return compile_proc.stdout.decode(), test.expected

            run_proc = subprocess.run(
                [str(exe_path)], input=extra["input_data"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, check=False,
            )
            output = "\n".join(part for part in (run_proc.stdout.strip(), run_proc.stderr.strip()) if part)
            if run_proc.returncode != 0:
                return output, test.expected

            if output == test.expected:
                return None
            return output, test.expected
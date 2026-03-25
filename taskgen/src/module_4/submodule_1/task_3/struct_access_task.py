from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem

_STRUCT_VARIANTS = [
    {
        "name": "Student",
        "struct_decl": "typedef struct { char name[32]; int age; float gpa; } Student;",
        "fields": ["name", "age", "gpa"],
        "target_field": "age",
        "target_type": "int",
        "printf_format": "%d"
    },
    {
        "name": "Book",
        "struct_decl": "typedef struct { char title[64]; int pages; double price; } Book;",
        "fields": ["title", "pages", "price"],
        "target_field": "pages",
        "target_type": "int",
        "printf_format": "%d"
    },
    {
        "name": "Point2D",
        "struct_decl": "typedef struct { double x; double y; int label; } Point2D;",
        "fields": ["x", "y", "label"],
        "target_field": "label",
        "target_type": "int",
        "printf_format": "%d"
    },
    {
        "name": "Rectangle",
        "struct_decl": "typedef struct { double width; double height; char color[16]; } Rectangle;",
        "fields": ["width", "height", "color"],
        "target_field": "width",
        "target_type": "double",
        "printf_format": "%.1f"
    },
    {
        "name": "Employee",
        "struct_decl": "typedef struct { char name[32]; int id; double salary; } Employee;",
        "fields": ["name", "id", "salary"],
        "target_field": "salary",
        "target_type": "double",
        "printf_format": "%.1f"
    }
]


class StructAccessTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % 5
        self.variant = _STRUCT_VARIANTS[self.variant_index]
        self.n_objects = 2 + (seed_value % 2)
        self.operation = seed_value % 3  # 0=max,1=sum,2=min

    def generate_task(self) -> str:
        op_text = ["максимум", "сумму", "минимум"][self.operation]
        read_order = ", ".join(self.variant["fields"])
        task_text = (
            "# Struct: доступ к полям\n\n"
            "### Задание №3\n\n"
            "- **Сложность:** средне  \n"
            "- **Формулировка:**  \n"
            f"  Вам дано объявление структуры:\n\n"
            f"  {self.variant['struct_decl']}\n\n"
            "  Напишите программу, которая:\n"
            f"  1. Считывает из стандартного ввода данные о {self.n_objects} объектах указанного типа "
            f"(каждая строка содержит значения полей в порядке: {', '.join(self.variant['fields'])}).\n"
            "  2. Сохраняет объекты в массив структур.\n"
            f"  3. Вычисляет и выводит {['максимум', 'сумму', 'минимум'][self.operation]} значений поля "
            f"{self.variant['target_field']}.\n\n"
            f"- Формат вывода: {self.variant['printf_format']}\n"
            "- Писать typedef структуры не нужно — только `main()`."
        )
        return task_text

    def _build_program_source(self) -> str:
        if self.solution is None:
            raise ValueError("Нет решения студента")
        student_code_path = Path(self.solution)
        if student_code_path.exists():
            student_code = student_code_path.read_text(encoding="utf-8")
        else:
            student_code = self.solution

        return textwrap.dedent(
            f"""
            #include <stdio.h>
            #include <string.h>

            {self.variant['struct_decl']}

            {student_code}
            """
        ).strip()

    def _generate_tests(self):
        self.tests = []
        import random
        random.seed(self.seed)
        for _ in range(1):
            inputs = []
            total = 0
            target_type = self.variant["target_type"]
            for _ in range(self.n_objects):
                vals = []
                for f in self.variant["fields"]:
                    if f in ("name", "title", "color"):
                        v = f"Val{random.randint(1,99)}"
                        vals.append(v)
                    elif f in ("age", "pages", "label", "id"):
                        v = random.randint(1, 100)
                        vals.append(str(v))
                    elif f in ("gpa", "price", "width", "height", "salary"):
                        v = round(random.uniform(1.0, 10.0), 1)
                        vals.append(str(v))
                inputs.append(" ".join(vals))

            target_values = []
            for line in inputs:
                parts = line.split()
                idx = self.variant["fields"].index(self.variant["target_field"])
                val = parts[idx]
                if target_type in ("int"):
                    target_values.append(int(val))
                else:
                    target_values.append(float(val))
            if self.operation == 0:
                result = max(target_values)
            elif self.operation == 1:
                result = sum(target_values)
            else:
                result = min(target_values)
            self.tests.append(
                TestItem(
                    input_str="\n".join(inputs),
                    showed_input=f"seed % 5 = {self.variant_index}",
                    expected=str(result),
                    compare_func=self._compare_default
                )
            )

    def compile(self) -> Optional[str]:
        return None

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
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
                return compile_proc.stdout.decode(), test.expected

            run_proc = subprocess.run(
                [str(exe_path)],
                input=test.input_str.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            output = run_proc.stdout.decode().strip()
            if run_proc.returncode != 0:
                return output, test.expected

            if self._compare_default(output, test.expected):
                return None
            return output, test.expected
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem
from src.module_4.submodule_1.task_1.struct_decl_task import _VARIANTS

# Строковые значения для разных полей
_STRING_VALUES = {
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"],
    "title": ["C_Primer", "Data_Structures", "Algorithms", "Python_Basics"],
    "color": ["red", "green", "blue", "yellow", "Diana"]
}


class StructInitTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self.seed_value = seed_value

    def _compute_field_values(self) -> dict[str, str]:
        val_dict = {}
        for field in self.variant.fields:
            if field.field_name in _STRING_VALUES:
                values = _STRING_VALUES[field.field_name]
                val_dict[field.field_name] = f'"{values[self.seed_value % len(values)]}"'
            elif field.c_type == "int":
                val_dict[field.field_name] = str(10 + self.seed_value % 90)
            elif field.c_type == "float":
                val_dict[field.field_name] = f"{(self.seed_value % 20) / 10.0 + 2.0:.1f}f"
            elif field.c_type == "double":
                val_dict[field.field_name] = f"{(self.seed_value % 100) / 10.0 + 1.0:.1f}"
        return val_dict

    def generate_task(self) -> str:
        val_dict = self._compute_field_values()
        field_lines = "\n".join(f"      - {field} = {value}" for field, value in val_dict.items())

        struct_decl = "; ".join(
            f"char {f.field_name}[{f.c_type[5:-1]}]" if f.c_type.startswith("char[") else f"{f.c_type} {f.field_name}"
            for f in self.variant.fields
        )

        task_text= (
            "# Struct: инициализация переменной\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            "  Вам дано объявление структуры:\n\n"
            f"  typedef struct {{ {struct_decl}; }} {self.variant.name};\n\n"
            f"  Объявите переменную `obj` структуры `{self.variant.name}` и инициализируйте её одним выражением при объявлении, используя указанные ниже значения полей.\n"
            "  Выведите все поля структуры через пробел в одну строку в формате, указанном в варианте.\n\n"
            "- Поля для инициализации:\n"
            f"{field_lines}\n\n"
            "  Подключать дополнительные заголовочные файлы (кроме `stdio.h`) не требуется.\n"
            "  Писать typedef структуры не нужно — только `main()`."
        )
        return task_text

    def _build_program_source(self) -> str:
        struct_decl_code = "; ".join(
            f"char {f.field_name}[{f.c_type[5:-1]}]" if f.c_type.startswith("char[") else f"{f.c_type} {f.field_name}"
            for f in self.variant.fields
        )

        if self.solution is None:
            raise ValueError("Нет решения студента")
        if Path(self.solution).exists():
            student_code = Path(self.solution).read_text(encoding="utf-8")
        else:
            student_code = self.solution

        return textwrap.dedent(
            f"""
            #include <stdio.h>
            #include <string.h>
            #include <assert.h>

            typedef struct {{ {struct_decl_code}; }} {self.variant.name};

            {student_code}
            """
        ).strip()

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        val_dict = self._compute_field_values()
        expected_output = " ".join(v.strip('"') for v in val_dict.values())
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"seed % {len(_VARIANTS)} = {self.variant_index} → {self.variant.name}",
                expected=expected_output,
                compare_func=self._compare_default,
            )
        ]

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

            run_proc = subprocess.run([str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            output = run_proc.stdout.decode().strip()
            if run_proc.returncode != 0:
                return output, test.expected

            if self._compare_default(output, test.expected):
                return None
            return output, test.expected
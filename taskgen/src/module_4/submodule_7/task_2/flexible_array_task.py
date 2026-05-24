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
    struct_def: str
    func_sig: str
    elem_type: str
    test_arrays: list


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        struct_def="typedef struct { int count; double values[]; } Stats;",
        func_sig="Stats *make_stats(double *arr, int n)",
        elem_type="double",
        test_arrays=[
            ("1.0 2.5 3.7", "3", "1.000000 2.500000 3.700000"),
            ("-1.5 0.0", "2", "-1.500000 0.000000"),
            ("42.42", "1", "42.420000"),
        ],
    ),
    1: VariantSpec(
        struct_def="typedef struct { int count; int items[]; } IntList;",
        func_sig="IntList *make_intlist(int *arr, int n)",
        elem_type="int",
        test_arrays=[
            ("10 20 30", "3", "10 20 30"),
            ("-5 0 42", "3", "-5 0 42"),
            ("100", "1", "100"),
        ],
    ),
    2: VariantSpec(
        struct_def="typedef struct { int count; float scores[]; } ScoreBoard;",
        func_sig="ScoreBoard *make_scoreboard(float *arr, int n)",
        elem_type="float",
        test_arrays=[
            ("1.5 2.25 3.0", "3", "1.500000 2.250000 3.000000"),
            ("-0.5 0.0", "2", "-0.500000 0.000000"),
            ("9.99", "1", "9.990000"),
        ],
    ),
    3: VariantSpec(
        struct_def="typedef struct { int count; char letters[]; } CharBuf;",
        func_sig="CharBuf *make_charbuf(char *arr, int n)",
        elem_type="char",
        test_arrays=[
            ("a b c", "3", "a b c"),
            ("x y", "2", "x y"),
            ("Z", "1", "Z"),
        ],
    ),
}


class FlexibleArrayTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Блок для самых умных: гибкие массивы\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            "  Вам дано объявление структуры с гибким массивом (соответствует вашему варианту).  \n"
            "  Напишите функцию `make_<name>`, которая:\n"
            "  1. Принимает массив элементов и их количество `n`.\n"
            "  2. Выделяет память под структуру с гибким массивом нужного размера через `malloc` (при неудачном `malloc` функция возвращает `NULL`).\n"
            "  3. Заполняет поле `count` значением `n` и копирует элементы в гибкий массив.\n"
            "  4. Возвращает указатель на созданную структуру.\n\n"
            "  Писать `main()` не нужно.\n\n"
            f"**Объявление структуры:**\n"
            f"```c\n{v.struct_def}\n```\n\n"
            f"**Сигнатура функции:** `{v.func_sig}`\n\n"
            f"Тип элементов гибкого массива: `{v.elem_type}`\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        for i, (input_str, n_str, expected_str) in enumerate(v.test_arrays):
            values = input_str.split()
            n = int(n_str)
            test = TestItem(
                input_str=f"arr = [{input_str}], n={n}",
                showed_input=f"make_{self._extract_struct_name(v.struct_def).lower()}(..., {n})",
                expected=f"OK:{n}::{expected_str}",
                compare_func=self._compare_default,
            )
            self.tests.append(test)
            self.test_extra.append({
                "values": values,
                "n": n,
                "expected_str": expected_str,
            })

    def _extract_struct_name(self, struct_def: str) -> str:
        match = re.search(r'}\s*(\w+)\s*;', struct_def)
        return match.group(1) if match else ""

    def _extract_flex_field(self, struct_def: str) -> str:
        match = re.search(r'\{\s*(.*?)\s*\}', struct_def, re.DOTALL)
        if not match:
            return "data"
        body = match.group(1)
        fields = [f.strip() for f in body.split(';') if f.strip()]
        for field in reversed(fields):
            if '[' in field and ']' in field:
                parts = field.split()
                return parts[-1].replace('[]', '')
        return "data"

    def _extract_func_name(self, func_sig: str) -> str:
        match = re.search(r'(\w+)\s*\(', func_sig)
        return match.group(1) if match else ""

    def _check_malloc_usage(self) -> bool:
        """Проверяет наличие вызова malloc (без проверки размера)."""
        code = re.sub(r'//.*', '', self.solution)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return bool(re.search(r'malloc\s*\(', code))

    def _check_malloc_size(self) -> bool:
        """
        Проверяет, что malloc выделяет память с учётом гибкого массива:
        malloc(sizeof(Struct) + n * sizeof(элемент))
        """
        v = self.variant
        struct_name = self._extract_struct_name(v.struct_def)
        elem_type = v.elem_type

        code = re.sub(r'//.*', '', self.solution)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        pattern = re.compile(
            r'malloc\s*\(\s*sizeof\s*\(\s*' + re.escape(struct_name) + r'\s*\)\s*\+\s*(?:n|count)\s*\*\s*sizeof\s*\(\s*' + re.escape(elem_type) + r'\s*\)\s*\)'
        )
        if pattern.search(code):
            return True

        pattern2 = re.compile(
            r'malloc\s*\(\s*sizeof\s*\(\s*' + re.escape(struct_name) + r'\s*\)\s*\+\s*\w+\s*\*\s*sizeof\s*\(\s*' + re.escape(elem_type) + r'\s*\)\s*\)'
        )
        if pattern2.search(code):
            return True

        return False

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        values = extra["values"]
        n = extra["n"]
        expected_str = extra["expected_str"]
        elem_type = v.elem_type

        if elem_type == "double":
            array_init = f"double test_arr[] = {{{', '.join(values)}}};"
            elem_printf = "%f"
        elif elem_type == "int":
            array_init = f"int test_arr[] = {{{', '.join(values)}}};"
            elem_printf = "%d"
        elif elem_type == "float":
            array_init = f"float test_arr[] = {{{', '.join(values)}}};"
            elem_printf = "%f"
        else:  # char
            quoted = [f"'{c}'" for c in values]
            array_init = f"char test_arr[] = {{{', '.join(quoted)}}};"
            elem_printf = "%c"

        struct_name = self._extract_struct_name(v.struct_def)
        func_name = self._extract_func_name(v.func_sig)
        flex_field = self._extract_flex_field(v.struct_def)

        check_code = f"""
    {array_init}
    {struct_name} *result = {func_name}(test_arr, {n});
    if (result == NULL) {{
        printf("FAIL: returned NULL\\n");
        return 1;
    }}
    if (result->count != {n}) {{
        printf("FAIL: count = %d, expected {n}\\n", result->count);
        free(result);
        return 1;
    }}
    for (int i = 0; i < {n}; i++) {{
        if (result->{flex_field}[i] != test_arr[i]) {{
            printf("FAIL: mismatch at index %d\\n", i);
            free(result);
            return 1;
        }}
    }}
    printf("OK:{n}::");
    for (int i = 0; i < {n}; i++) {{
        if (i > 0) printf(" ");
        printf("{elem_printf}", result->{flex_field}[i]);
    }}
    printf("\\n");
    free(result);
    return 0;
"""
        program = textwrap.dedent(f"""
        #include <stdio.h>
        #include <stdlib.h>

        {v.struct_def}

        {self.solution}

        int main() {{
            {check_code}
        }}
        """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if not self._check_malloc_size():
            return "Ошибка: память выделяется неправильно.", "требуется правильный размер malloc"

        if not self._check_malloc_usage():
            return "Ошибка: не найден вызов malloc() для выделения памяти", "требуется malloc"

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
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, check=False,
            )
            output = "\n".join(part for part in (run_proc.stdout.decode().strip(), run_proc.stderr.decode().strip()) if part)
            if run_proc.returncode != 0:
                return output, test.expected

            if output.startswith(test.expected):
                return None
            return output, test.expected
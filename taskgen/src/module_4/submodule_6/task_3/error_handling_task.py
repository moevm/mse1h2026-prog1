from dataclasses import dataclass
from typing import Optional
import subprocess
import tempfile
import textwrap
from pathlib import Path

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    step1_name: str
    step1_params: str
    step1_return_type: str
    step2_name: str
    step2_params: str
    step2_return_type: str
    process_signature: str
    process_call_args: str
    result_type: str
    default_input: str
    test_cases: list


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        step1_name="parse_int",
        step1_params='s, &tmp',
        step1_return_type="int",
        step2_name="double_val",
        step2_params='tmp, out',
        step2_return_type="int",
        process_signature="int process(const char *s, int *out)",
        process_call_args='"123", &result',
        result_type="int",
        default_input='"123"',
        test_cases=[
            (0, 0, 123, 246, 0, "246"),
            (1, 0, 0, 0, 1, ""),
            (0, 1, 123, 0, 2, ""),
        ],
    ),
    1: VariantSpec(
        step1_name="read_positive",
        step1_params='x, &tmp',
        step1_return_type="int",
        step2_name="safe_sqrt",
        step2_params='tmp, out',
        step2_return_type="int",
        process_signature="int process(int x, int *out)",
        process_call_args='10, &result',
        result_type="int",
        default_input='10',
        test_cases=[
            (0, 0, 10, 3, 0, "3"),
            (1, 0, 0, 0, 1, ""),
            (0, 1, 10, 0, 2, ""),
        ],
    ),
    2: VariantSpec(
        step1_name="clamp",
        step1_params='x, limit, &tmp',
        step1_return_type="int",
        step2_name="invert",
        step2_params='tmp, out',
        step2_return_type="int",
        process_signature="int process(int x, int limit, int *out)",
        process_call_args='42, 100, &result',
        result_type="int",
        default_input='42, 100',
        test_cases=[
            (0, 0, 42, -42, 0, "-42"),
            (1, 0, 0, 0, 1, ""),
            (0, 1, 42, 0, 2, ""),
        ],
    ),
    3: VariantSpec(
        step1_name="parse_double",
        step1_params='s, &tmp',
        step1_return_type="int",
        step2_name="safe_log",
        step2_params='tmp, out',
        step2_return_type="int",
        process_signature="int process(const char *s, double *out)",
        process_call_args='"2.71828", &result',
        result_type="double",
        default_input='"2.71828"',
        test_cases=[
            (0, 0, 2.71828, 1.0, 0, "1.000000"),
            (1, 0, 0.0, 0.0, 1, ""),
            (0, 1, 2.71828, 0.0, 2, ""),
        ],
    ),
}


class ErrorHandlingTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Обработка ошибок: возврат кодов ошибок\n\n"
            "### Задание №3\n\n"
            "- **Формулировка:**  \n"
            "  Вам даны объявления двух функций и их коды возврата (соответствует вашему варианту).  \n"
            "  Напишите функцию-обёртку `process`, которая:\n"
            "  1. Вызывает первую функцию (`step1`). Если та вернула ненулевой код — немедленно возвращает `1`.\n"
            "  2. Вызывает вторую функцию (`step2`) с результатом первой. Если та вернула ненулевой код — немедленно возвращает `2`.\n"
            "  3. Если обе функции завершились успешно — записывает итоговый результат в `*out` и возвращает `0`.\n\n"
            "  Писать `main()` не нужно.\n\n"
            f"**Сигнатуры функций step1 и step2:**\n"
            f"- `{v.step1_return_type} {v.step1_name}({self._signature_params(v.step1_params)})`\n"
            f"- `{v.step2_return_type} {v.step2_name}({self._signature_params(v.step2_params)})`\n"
            f"**Сигнатура process:** `{v.process_signature}`\n\n"
            "**Коды возврата:** `0` — успех, любое другое значение — ошибка.\n"
            "**Коды возврата process:** `0` — успех, `1` — ошибка на шаге 1, `2` — ошибка на шаге 2.\n"
        )

    def _signature_params(self, params: str) -> str:
        if 's, &tmp' in params:
            return "const char *s, int *tmp"
        if 'x, &tmp' in params:
            return "int x, int *tmp"
        if 'x, limit, &tmp' in params:
            return "int x, int limit, int *tmp"
        if 's, &tmp' in params and self.variant.result_type == "double":
            return "const char *s, double *tmp"
        if 'tmp, out' in params:
            if self.variant.result_type == "double":
                return "double tmp, double *out"
            else:
                return "int tmp, int *out"
        return "?"

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        for i, tc in enumerate(v.test_cases):
            step1_ret, step2_ret, step1_out, step2_out, expected_ret, expected_out = tc
            test = TestItem(
                input_str=f"test_{i}",
                showed_input=f"process({v.process_call_args})",
                expected=f"RET={expected_ret} OUT={expected_out}",
                compare_func=self._compare_default,
            )
            self.tests.append(test)
            self.test_extra.append({
                "step1_ret": step1_ret,
                "step2_ret": step2_ret,
                "step1_out": str(step1_out),
                "step2_out": str(step2_out),
                "expected_ret": expected_ret,
                "expected_out": expected_out,
            })

    def _build_test_program(self, extra: dict) -> str:
        v = self.variant
        step1_ret = extra["step1_ret"]
        step2_ret = extra["step2_ret"]
        step1_out = extra["step1_out"]
        step2_out = extra["step2_out"]

        if self.variant_index == 0:
            step1_def = f"""
int {v.step1_name}(const char *s, int *out) {{
    if ({step1_ret} != 0) return {step1_ret};
    *out = {step1_out};
    return 0;
}}"""
        elif self.variant_index == 1:
            step1_def = f"""
int {v.step1_name}(int x, int *out) {{
    if ({step1_ret} != 0) return {step1_ret};
    *out = {step1_out};
    return 0;
}}"""
        elif self.variant_index == 2:
            step1_def = f"""
int {v.step1_name}(int x, int limit, int *out) {{
    if ({step1_ret} != 0) return {step1_ret};
    *out = {step1_out};
    return 0;
}}"""
        else:
            step1_def = f"""
int {v.step1_name}(const char *s, double *out) {{
    if ({step1_ret} != 0) return {step1_ret};
    *out = {step1_out};
    return 0;
}}"""

        if v.result_type == "int":
            step2_def = f"""
int {v.step2_name}(int x, int *out) {{
    if ({step2_ret} != 0) return {step2_ret};
    *out = {step2_out};
    return 0;
}}"""
        else:
            step2_def = f"""
int {v.step2_name}(double x, double *out) {{
    if ({step2_ret} != 0) return {step2_ret};
    *out = {step2_out};
    return 0;
}}"""

        if v.result_type == "int":
            out_var = "int result = 0;"
            print_code = """
    if (ret == 0) {
        printf("RET=0 OUT=%d\\n", result);
    } else {
        printf("RET=%d OUT=\\n", ret);
    }
"""
        else:
            out_var = "double result = 0.0;"
            print_code = """
    if (ret == 0) {
        printf("RET=0 OUT=%.6f\\n", result);
    } else {
        printf("RET=%d OUT=\\n", ret);
    }
"""

        main_code = f"""
    {out_var}
    int ret = process({v.process_call_args});
    {print_code}
    return 0;
"""

        program = textwrap.dedent(f"""
        #include <stdio.h>

        {step1_def}

        {step2_def}

        {self.solution}

        int main() {{
            {main_code}
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
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, check=False,
            )
            output = "\n".join(part for part in (run_proc.stdout.decode().strip(), run_proc.stderr.decode().strip()) if part)
            if run_proc.returncode != 0:
                return output, test.expected

            if output == test.expected:
                return None
            return output, test.expected
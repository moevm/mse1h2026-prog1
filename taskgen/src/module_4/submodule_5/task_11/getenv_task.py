from dataclasses import dataclass
from typing import Optional
import subprocess
import tempfile
import textwrap
import os
import re
from pathlib import Path

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    return_type: str
    env_name: str
    default_val: str
    test_cases: list   # (env_value, expected)


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        return_type="int",
        env_name="TIMEOUT",
        default_val="30",
        test_cases=[
            (None, "30"),
            ("10", "10"),
            ("abc", "0"),
            ("  42  ", "42"),
            ("-5", "-5"),
        ],
    ),
    1: VariantSpec(
        return_type="double",
        env_name="THRESHOLD",
        default_val="0.5",
        test_cases=[
            (None, "0.500000"),
            ("1.25", "1.250000"),
            ("abc", "0.000000"),
            ("  -3.7  ", "-3.700000"),
        ],
    ),
    2: VariantSpec(
        return_type="long",
        env_name="MAX_SIZE",
        default_val="1024",
        test_cases=[
            (None, "1024"),
            ("2048", "2048"),
            ("2000000000", "2000000000"),
            ("xyz", "0"),
        ],
    ),
}


class GetEnvTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        if v.return_type == "int":
            convert_func = "atoi"
        elif v.return_type == "double":
            convert_func = "atof"
        else:
            convert_func = "atol"

        return (
            "# stdlib: getenv\n\n"
            "### Задание №11\n\n"
            "- **Формулировка:**  \n"
            f"  Напишите функцию `{v.return_type} get_env_value(const char *name, {v.return_type} default_val)`.\n"
            f"  Она должна считывать переменную окружения (имя передаётся в параметре `name`),\n"
            f"  преобразовывать её строковое значение в `{v.return_type}` с помощью `{convert_func}`\n"
            "  и возвращать результат. Если переменная не задана — возвращать `default_val`.\n\n"
            "  Писать `main()` не нужно.\n"
            "  Подключать заголовочные файлы не требуется.\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _check_conversion_function(self) -> bool:
        """Проверяет, что используется именно требуемая функция преобразования."""
        v = self.variant
        expected_func = ""
        if v.return_type == "int":
            expected_func = "atoi"
        elif v.return_type == "double":
            expected_func = "atof"
        else:
            expected_func = "atol"
        
        lines = self.solution.split('\n')
        code_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            code_lines.append(line)
        code = ' '.join(code_lines)
        
        pattern = r'\b' + re.escape(expected_func) + r'\s*\('
        return bool(re.search(pattern, code))

    def _generate_tests(self):
        v = self.variant
        self.tests = []
        self.test_extra = []

        for env_val, expected_str in v.test_cases:
            test = TestItem(
                input_str=f"env[{v.env_name}] = '{env_val}'",
                showed_input=f"get_env_value(\"{v.env_name}\", {v.default_val})",
                expected=f"OK:{expected_str}",
                compare_func=self._compare_default,
            )
            self.tests.append(test)
            self.test_extra.append({
                "env_value": env_val,
                "expected": expected_str,
            })

    def _build_test_program(self) -> str:
        v = self.variant
        if v.return_type == "double":
            check_code = f"""
    double result = get_env_value("{v.env_name}", {v.default_val});
    printf("OK:%.6f\\n", result);
    return 0;
"""
        elif v.return_type == "long":
            check_code = f"""
    long result = get_env_value("{v.env_name}", {v.default_val});
    printf("OK:%ld\\n", result);
    return 0;
"""
        else:  # int
            check_code = f"""
    int result = get_env_value("{v.env_name}", {v.default_val});
    printf("OK:%d\\n", result);
    return 0;
"""

        program = textwrap.dedent(f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>

        {self.solution}

        int main() {{
            {check_code}
        }}
        """)
        return program

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if not self._check_conversion_function():
            return f"Ошибка: должна использоваться функция {self.variant.return_type == 'int' and 'atoi' or self.variant.return_type == 'double' and 'atof' or 'atol'}", "требуется правильная функция преобразования"

        try:
            idx = self.tests.index(test)
        except ValueError:
            return "Test not found", "unknown"
        extra = self.test_extra[idx]

        program_source = self._build_test_program()
        env_value = extra["env_value"]
        expected = extra["expected"]

        env = os.environ.copy()
        if env_value is None:
            env.pop(self.variant.env_name, None)
        else:
            env[self.variant.env_name] = env_value

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
                cwd=tmpdir, env=env, check=False,
            )
            output = "\n".join(part for part in (run_proc.stdout.decode().strip(), run_proc.stderr.decode().strip()) if part)
            if run_proc.returncode != 0:
                return output, test.expected

            if output == test.expected:
                return None
            return output, test.expected
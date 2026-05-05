from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task4(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> tuple[str, str, str, int]:
        rem3 = self.seed % 3
        func_names = ["make_absolute", "normalize_value", "get_abs_ref"]
        param_names = ["ptr", "val_ptr", "num"]
        fmt_strings = [
            "Result: %d\n",
            "Модуль => %d\n",
            "[ABS] %d\n",
            "Absolute value: %d\n"
        ]

        func_name = func_names[rem3]
        param_name = param_names[rem3]
        fmt_str = fmt_strings[self.seed % 4]
        test_val = -(self.seed % 100)
        if test_val == 0: test_val = -5  

        return func_name, param_name, fmt_str, test_val

    def _get_test_driver_code(self) -> str:
        func, param, _, val = self._get_params()
        return f'''#include <stdio.h>

void {func}(int *{param});

int main() {{
    int val = {val};
    {func}(&val);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func, param, fmt, _ = self._get_params()
        return f"""### Тема: Разыменовывание

**Сложность:** легкая

**Задание:**
Напишите функцию `void {func}(int *{param})`, которая изменяет значение по адресу `{param}` на его модуль (абсолютное значение). После изменения выведите полученное значение в формате:
{fmt}
"""

    def _generate_tests(self):
        _, _, fmt_str, test_val = self._get_params()
        expected_val = abs(test_val)
        expected = fmt_str.replace("%d", str(expected_val))

        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"*ptr = {test_val}",
                expected=expected,
                compare_func=self._compare_default
            )
        ]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func, param, _, _ = self._get_params()

        sig_pattern = rf'void\s+{func}\s*\(\s*int\s*\*\s*{param}\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: неверная сигнатура функции `{func}(int *{param})`."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
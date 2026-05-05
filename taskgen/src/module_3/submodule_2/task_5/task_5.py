from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task5(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> tuple[str, str, str, str, str]:
        rem3 = self.seed % 3
        func_names = ["sum_via_ptrs", "add_pointed_vals", "compute_sum_ref"]
        p1_names = ["a", "lhs", "val_x"]
        p2_names = ["b", "rhs", "val_y"]
        p3_names = ["res", "out", "target"]
        fmts = ["Calculated: %d\n", "Результат => %d\n", "[Output] %d\n", "Final value -> %d\n"]

        return (
            func_names[rem3],
            p1_names[rem3],
            p2_names[rem3],
            p3_names[rem3],
            fmts[self.seed % 4]
        )

    def _get_test_driver_code(self) -> str:
        func, p1, p2, p3, _ = self._get_params()
        return f'''#include <stdio.h>

void {func}(int *{p1}, int *{p2}, int *{p3});

int main() {{
    int x, y, z;
    if (scanf("%d %d", &x, &y) != 2) return 1;
    {func}(&x, &y, &z);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func, p1, p2, p3, fmt = self._get_params()
        return f"""### Тема: Разыменовывание

**Сложность:** легкая

**Задание:**
Напишите функцию `{func}(int *{p1}, int *{p2}, int *{p3})`, которая разыменовывает указатели `{p1}` и `{p2}`, вычисляет сумму значений и записывает результат по адресу `{p3}`. Внутри функции выведите полученное значение в формате `{fmt}`.
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        _, _, _, _, fmt_str = self._get_params()

        base_pairs = [(10, 7), (-5, 12), (0, 0), (100, -50), (-15, -20)]
        for _ in range(self.tests_num - len(base_pairs)):
            base_pairs.append((random.randint(-20, 50), random.randint(-20, 50)))

        for v1, v2 in base_pairs[:self.tests_num]:
            input_str = f"{v1} {v2}"
            expected = fmt_str.replace("%d", str(v1 + v2))
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"*p1={v1}, *p2={v2}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func, p1, p2, p3, _ = self._get_params()

        sig_pattern = rf'void\s+{func}\s*\(\s*int\s*\*\s*{p1}\s*,\s*int\s*\*\s*{p2}\s*,\s*int\s*\*\s*{p3}\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: неверная сигнатура функции `{func}(int *{p1}, int *{p2}, int *{p3})`."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
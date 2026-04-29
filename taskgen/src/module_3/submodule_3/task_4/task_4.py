from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule3_Task4(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self):
        variant = self.seed % 4
        configs = [
            ("print_dual_arr",    "i * 2 + 1",    lambda i: i * 2 + 1,      "Idx[%d]: %d | Ptr: %d\n"),
            ("show_idx_vs_ptr",   "i * i",        lambda i: i * i,          "El[%d]: %d | Ptr: %d\n"),
            ("array_dual_output", "100 - i * 3",  lambda i: 100 - i * 3,    "Arr[%d]: %d | Ptr: %d\n"),
            ("demo_access_modes", "i + 10",       lambda i: i + 10,         "Out[%d]: %d | Ptr: %d\n")
        ]
        return configs[variant]

    def _get_test_driver_code(self) -> str:
        func_name, _, _, _ = self._get_params()
        return f'''#include <stdio.h>

void {func_name}(int size);

int main() {{
    int size;
    if (scanf("%d", &size) != 1) return 1;
    if (size <= 0 || size > 20) return 1;
    {func_name}(size);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func_name, init_formula, _, print_format = self._get_params()
        fmt_escaped = print_format.replace('\n', '\\n')
        return f"""### Тема: Массивы vs Указатели 
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {func_name}(int size)`, которая объявляет массив целых чисел `int arr[size]` и инициализирует его элементы по формуле: `arr[i] = {init_formula}`. Выведите все элементы массива, используя два способа доступа: через индекс `arr[i]` и через арифметику указателей.

**Формат вывода:** `{print_format}`
"""
    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        _, _, calc, print_format = self._get_params()

        sizes = [1, 3, 5, 10]
        for _ in range(self.tests_num - len(sizes)):
            sizes.append(random.randint(2, 15))

        for size in sizes:
            expected = ""
            for i in range(size):
                val = calc(i)
                expected += print_format % (i, val, val)

            self.tests.append(TestItem(
                input_str=str(size),
                showed_input=f"size={size}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func_name, init_formula, _, print_format = self._get_params()

        if not re.search(rf'void\s+{func_name}\s*\(\s*int\s+size\s*\)', self.solution):
            return f"Ошибка: неверная сигнатура функции `{func_name}(int size)`."

        if not re.search(r'\*\s*\(?\s*arr\s*\+|\*\s*ptr\s*\+\+|ptr\s*=\s*arr', self.solution):
            return "Ошибка: не найдена арифметика указателей (разрешено: `*(arr + i)`, `*ptr++` и т.п.)."

        fmt_escaped = re.escape(print_format.replace('\n', '\\n'))
        if not re.search(rf'printf\s*\(.*{fmt_escaped}.*arr\s*\[.*\].*\*.*arr.*\)', self.solution, re.DOTALL):
            return f"Ошибка: printf должен использовать формат и выводить значения через arr[i] и *(arr + i)."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
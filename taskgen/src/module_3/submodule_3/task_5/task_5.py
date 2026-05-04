from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule3_Task5(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {
            "test_driver.c": self._get_test_driver_code(),
        }

    def _get_test_driver_code(self) -> str:
        func_name = "calc_array_sum" if self.seed % 2 == 0 else "ptr_sum_elements"
        return f'''#include <stdio.h>

int {func_name}(int *arr, int size);

int main() {{
    int size;
    if (scanf("%d", &size) != 1 || size <= 0 || size > 100) return 1;
    
    int arr[100];
    for (int i = 0; i < size; i++) {{
        scanf("%d", &arr[i]);
    }}
    
    {func_name}(arr, size);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func_name = "calc_array_sum" if self.seed % 2 == 0 else "ptr_sum_elements"
        print_format = "Sum: %d\\n" if self.seed % 2 == 0 else "Total sum: %d\\n"
        
        return f"""### Тема: Передача массива в функцию

**Сложность:** средняя

**Задание:**
Напишите функцию `int {func_name}(int *arr, int size)`, которая принимает указатель на первый элемент массива и количество элементов. Используйте только арифметику указателей `*(arr + i)` для доступа к данным. Вычислите сумму всех элементов и выведите её в консоль, строго соблюдая формат `{print_format}`. 
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        print_format = "Sum: %d\n" if self.seed % 2 == 0 else "Total sum: %d\n"

        base_cases = [
            [3, 10, 20, 30],
            [1, 42],
            [4, -5, 0, 15, -10]
        ]
        for _ in range(self.tests_num - len(base_cases)):
            size = random.randint(2, 8)
            elems = [random.randint(-100, 100) for _ in range(size)]
            base_cases.append([size] + elems)

        for case in base_cases:
            size = case[0]
            elems = case[1:]
            input_str = " ".join(map(str, case))
            expected_sum = sum(elems)
            expected = print_format % expected_sum
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"arr = {elems}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func_name = "calc_array_sum" if self.seed % 2 == 0 else "ptr_sum_elements"

        if f"{func_name}(int *arr, int size)" not in self.solution:
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if re.search(r'\b\w+\s*\[', self.solution):
            return "Ошибка: запрещено использовать синтаксис индексации [...]. Используйте только арифметику указателей *(arr + i)."

        if not re.search(r'\*\s*\(\s*\w+\s*\+\s*\w+\s*\)', self.solution):
            return "Ошибка: не найдено использование арифметики указателей *(ptr + i)."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
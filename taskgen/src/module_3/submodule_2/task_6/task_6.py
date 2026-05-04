from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM

class Module3_Submodule2_Task6(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {
            "tests_num": DEFAULT_TEST_NUM
        }
        default_params.update(kwargs)

        super().__init__(**default_params)
        self.check_files = {
            "test_driver.c": self._get_test_driver_code(),
        }

    def _get_test_driver_code(self) -> str:
        if self.seed % 2 == 0:
            return r'''#include <stdio.h>

void change_value(float *ptr, float new_value);

int main() {
    float value, new_val;
    if (scanf("%f %f", &value, &new_val) != 2) {
        return 1;
    }
    change_value(&value, new_val);
    return 0;
}
'''
        else:
            return r'''#include <stdio.h>

void transform_number(int *ptr, int new_value);

int main() {
    int value, new_val;
    if (scanf("%d %d", &value, &new_val) != 2) {
        return 1;
    }
    transform_number(&value, new_val);
    return 0;
}
'''

    def generate_task(self) -> str:
        if self.seed % 2 == 0:
            return """### Тема: Адрес переменной 

**Сложность:** легкая

**Задача:**
Напишите функцию `void change_value(float *ptr, float new_value)`, которая:
1. Выведет значение `*ptr` до изменения.
2. Изменит значение по адресу `ptr` на `new_value`.
3. Выведет значение `*ptr` после изменения.

Формат вывода:
Value changed from `old_val` to `new_val`
"""
        else:
            return """### Тема: Адрес переменной 

**Сложность:** легкая

**Задача:**
Напишите функцию `void transform_number(int *ptr, int new_value)`, которая:
1. Выведет значение `*ptr` до изменения.
2. Изменит значение по адресу `ptr` на `new_value`.
3. Выведет значение `*ptr` после изменения.

Формат вывода:
Before: `old_value`
After: `new_value`
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        if self.seed % 2 == 0:  
            test_cases = [
                (5.0, 10.0), (0.0, 100.0), (-5.0, 20.0), (42.0, 0.0), (-10.0, -20.0)
            ]
            for _ in range(self.tests_num - len(test_cases)):
                test_cases.append((
                    float(random.randint(-1000, 1000)),
                    float(random.randint(-1000, 1000))
                ))

            for old_val, new_val in test_cases:
                expected = f"Value changed from {old_val:g} to {new_val:g}\n"
                input_str = f"{old_val:g} {new_val:g}"

                self.tests.append(TestItem(
                    input_str=input_str,
                    showed_input=f"*ptr = {old_val:g}, new_value = {new_val:g}",
                    expected=expected,
                    compare_func=self._compare_default
                ))
        else: 
            test_cases = [
                (5, 10), (0, 100), (-5, 20), (42, 0), (-10, -20)
            ]
            for _ in range(self.tests_num - len(test_cases)):
                test_cases.append((
                    random.randint(-1000, 1000),
                    random.randint(-1000, 1000)
                ))

            for old_val, new_val in test_cases:
                expected = f"Before: {old_val}\nAfter: {new_val}\n"
                input_str = f"{old_val} {new_val}"

                self.tests.append(TestItem(
                    input_str=input_str,
                    showed_input=f"*ptr = {old_val}, new_value = {new_val}",
                    expected=expected,
                    compare_func=self._compare_default
                ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if self.seed % 2 == 0:
            if "void change_value(float *ptr, float new_value)" not in self.solution:
                return "Ошибка: функция change_value имеет неверную сигнатуру или отсутствует."
        else:
            if "void transform_number(int *ptr, int new_value)" not in self.solution:
                return "Ошибка: функция transform_number имеет неверную сигнатуру или отсутствует."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        
        return normalize(output) == normalize(expected)
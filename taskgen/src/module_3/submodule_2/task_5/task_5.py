from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task5(BaseTaskClass):
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

void solution(int *ptr, int new_value);

int main() {
    int value, new_val;
    if (scanf("%d %d", &value, &new_val) != 2) {
        return 1;
    }
    solution(&value, new_val);
    return 0;
}
'''
        else:
            return r'''#include <stdio.h>
#include <stddef.h>

void print_value(int *ptr);

int main() {
    int value;
    scanf("%d", &value) ;

    if (value == 0){
        print_value(NULL);
    }

    else{
        print_value(&value);
    }
    return 0;
}
'''
    
    def generate_task(self) -> str:
        if not self.seed % 2 == 0:
            return """
            Изменение значения через указатель.

            Напишите функцию void solution(int *ptr, int new_value), которая:
            1. Выведет значение *ptr до изменения
            2. Изменит значение по адресу ptr на new_value
            3. Выведет значение *ptr после изменения

            Формат вывода:
            Before: X
            After: Y
            """
        else:
            return """
            Чтение переменной через указатель

            Напишите функцию void print_value(int *ptr), которая принимает указатель на целое число 
            и выводит значение, находящееся по этому адресу. 
            Если указатель NULL вывести Value: 0

            Формат вывода:
            Value: X
            """

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        if self.seed % 2 == 0:
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
        else:
            test_values = [5, 0, -5, 42, -10]

            for _ in range(self.tests_num - len(test_values)):
                test_values.append(random.randint(-1000, 1000))
            
            for val in test_values:
                expected = f"Value: {val}\n"
                input_str = f"{val}"

                if val == 0:
                    showed_input = "*ptr = NULL"
                else:
                    showed_input = f"*ptr = {val}"
                
                self.tests.append(TestItem(
                    input_str=input_str,
                    showed_input=showed_input,
                    expected=expected,
                    compare_func=self._compare_default
                ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if self.seed % 2 == 0:
            if "void solution(int *ptr, int new_value)" not in self.solution:
                return "Ошибка: функция solution имеет неверную сигнатуру или отсутствует."
        else:
            if "void print_value(int *ptr)" not in self.solution:
                return "Ошибка: функция print_value имеет неверную сигнатуру или отсутствует."
            if "NULL" not in self.solution and "nullptr" not in self.solution:
                return "Ошибка: не найдена проверка указателя на NULL."
        
        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        
        return normalize(output) == normalize(expected)
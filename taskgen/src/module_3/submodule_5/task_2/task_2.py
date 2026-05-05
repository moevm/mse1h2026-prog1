from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task2(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_test_driver_code(self) -> str:
        func_name = "init_calloc_array" if self.seed % 2 == 0 else "process_calloc_buf"
        return f'''#include <stdio.h>
#include <stdlib.h>

void {func_name}(int size);

int main() {{
    int size;
    if (scanf("%d", &size) != 1) return 1;
    {func_name}(size);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func_name = "init_calloc_array" if self.seed % 2 == 0 else "process_calloc_buf"
        type_str = "int" if self.seed % 2 == 0 else "float"
        formula = f"({self.seed} + i) + 1" if self.seed % 2 == 0 else f"({self.seed} + i) * 1.5"
        
        return f"""### Тема: "Функция calloc()"
**Сложность:** средняя
**Задание:**
Создайте функцию `{func_name}(int size)`, которая выделяет память с помощью `calloc` для массива из `size` элементов типа `{type_str}`. Инициализируйте массив значениями по формуле: `arr[i] = {formula}`. Выведите все элементы массива в формате `Array: val1, val2, ..., valN`. Освободите память с помощью `free()`. Проверка результата `calloc` на `NULL` обязательна: если выделение памяти оказалось неудачным, выведите `Allocation failed` в `stderr` и завершите работу функции.
Формат вывода:
`Array: val1, val2, ..., valN`
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        is_int = (self.seed % 2 == 0)
        base_val = self.seed  
        sizes = [1, 3, 5, 10]
        for _ in range(self.tests_num - len(sizes)):
            sizes.append(random.randint(2, 20))

        for size in sizes:
            input_str = str(size)
            if size <= 0:
                expected = ""
            else:
                vals = []
                for i in range(size):
                    if is_int:
                        val = int(base_val + i + 1)
                    else:
                        val = (base_val + i) * 1.5
                    vals.append(val)

                formatted = [str(v) for v in vals] if is_int else [f"{v:g}" for v in vals]
                expected = "Array: " + ", ".join(formatted) + "\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"size={size}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func_name = "init_calloc_array" if self.seed % 2 == 0 else "process_calloc_buf"

        if f"{func_name}(int size)" not in self.solution:
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."
        if "calloc" not in self.solution:
            return "Ошибка: не найдено использование calloc."
        if "free" not in self.solution:
            return "Ошибка: не найдено использование free."

        if "NULL" not in self.solution and "nullptr" not in self.solution:
            return "Ошибка: не найдена проверка результата calloc на NULL. Используйте конструкцию if (arr == NULL) { ... }"

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule5_Task3(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_test_driver_code(self) -> str:
        func_name = "realloc_int" if self.seed % 2 == 0 else "realloc_float"
        return f'''#include <stdio.h>
#include <stdlib.h>

void {func_name}(int initial_size, int new_size);

int main() {{
    int i_size, n_size;
    if (scanf("%d %d", &i_size, &n_size) != 2) return 1;
    {func_name}(i_size, n_size);
    return 0;
}}
'''

    def generate_task(self) -> str:
        func_name = "realloc_int" if self.seed % 2 == 0 else "realloc_float"
        initial_func = "malloc" if self.seed % 2 == 0 else "calloc"
        type_str = "int" if self.seed % 2 == 0 else "float"
        init_f = f"({self.seed} + i) + 1" if self.seed % 2 == 0 else f"({self.seed} + i) * 1.5"
        new_f = f"({self.seed} + i) * 2" if self.seed % 2 == 0 else f"({self.seed} + i) + 10.0"

        return f"""### Тема: "Функция realloc()"
**Сложность:** средняя
**Задание:**
Создайте функцию `{func_name}(int initial_size, int new_size)`, которая выделяет память для массива из `initial_size` элементов типа `{type_str}` с помощью `{initial_func}`. Инициализируйте существующие элементы значениями по формуле: `arr[i] = {init_f}`. Измените размер массива на `new_size` элементов с помощью `realloc`. Заполните *только новые* элементы значениями по формуле: `arr[i] = {new_f}`. Выведите все элементы итогового массива в формате `Array: val1, val2, ..., valN`. Освободите память с помощью `free()`. Проверка результатов выделения и изменения памяти на `NULL` обязательна: при ошибке `realloc` освободите исходный блок, выведите `Allocation failed` и завершите работу функции. 
Формат вывода:
`Array: val1, val2, ..., valN`
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []
        is_float = (self.seed % 2 == 1)
        s = self.seed
        cases = [
            (3, 6), (1, 5), (5, 5), (2, 8), (4, 10)
        ]
        for _ in range(self.tests_num - len(cases)):
            i_s = random.randint(1, 8)
            n_s = random.randint(i_s, i_s + 7)
            cases.append((i_s, n_s))

        for i_size, n_size in cases:
            input_str = f"{i_size} {n_size}"
            
            if i_size <= 0 or n_size <= 0:
                expected = ""
            else:
                vals = []
                for i in range(n_size):
                    if i < i_size:
                        val = (s + i) + 1 if not is_float else (s + i) * 1.5
                    else:
                        val = (s + i) * 2 if not is_float else (s + i) + 10.0
                    vals.append(val)

                formatted = [str(int(v)) for v in vals] if not is_float else [f"{v:g}" for v in vals]
                expected = "Array: " + ", ".join(formatted) + "\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f"initial_size={i_size}, new_size={n_size}",
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        func_name = "realloc_int" if self.seed % 2 == 0 else "realloc_float"
        if f"{func_name}(int initial_size, int new_size)" not in self.solution:
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."
        if "realloc" not in self.solution:
            return "Ошибка: не найдено использование realloc."
        if "free" not in self.solution:
            return "Ошибка: не найдено использование free."

        if "NULL" not in self.solution and "nullptr" not in self.solution:
            return "Ошибка: не найдена проверка результата выделения памяти на NULL."

        initial_func = "malloc" if self.seed % 2 == 0 else "calloc"
        if initial_func not in self.solution:
            return f"Ошибка: в вашем варианте требуется использовать {initial_func} для начального выделения."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
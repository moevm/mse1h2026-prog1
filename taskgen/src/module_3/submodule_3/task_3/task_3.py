from typing import Optional
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule3_Task3(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}
        self._sizes = {"int": 4, "float": 4, "char": 1, "ptr": 8}

    def _get_params(self) -> tuple[str, str, int]:
        rem = self.seed % 3
        if rem == 0: func, typ = "show_sizes_int", "int"
        elif rem == 1: func, typ = "show_sizes_float", "float"
        else: func, typ = "show_sizes_char", "char"
        return func, typ, max(1, self.seed % 30)

    def _get_test_driver_code(self) -> str:
        func, _, size = self._get_params()
        return f'''#include <stdio.h>
void {func}(int size);
int main() {{
    {func}({size});
    return 0;
}}
'''

    def generate_task(self) -> str:
        func, typ, size = self._get_params()
        return f"""### Тема: Сравнение размеров (указатель vs массив)
**Сложность:** средняя
**Задание:**
Напишите функцию `void {func}(int size)`, которая объявляет массив `{typ} arr[size]` и указатель на его начало. Выведите размер массива в байтах, размер указателя и размер одного элемента массива, строго соблюдая заданный формат. Запрещено использовать глобальные переменные и библиотеки кроме `<stdio.h>`. Для вычисления размеров используйте оператор `sizeof`.
"""

    def _generate_tests(self):
        _, typ, size = self._get_params()
        e = self._sizes[typ]
        expected = f"Array size: {e * size}\nPointer size: {self._sizes['ptr']}\nElement size: {e}\n"
        self.tests = [TestItem(
            input_str="",
            showed_input=f"type={typ}, size={size}",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err: return err
        func, typ, _ = self._get_params()
        if not re.search(rf'void\s+{func}\s*\(\s*int\s+size\s*\)', self.solution):
            return f"Ошибка: функция {func} имеет неверную сигнатуру или отсутствует."
        if 'sizeof' not in self.solution:
            return "Ошибка: необходимо использовать оператор sizeof для вычисления размеров."
        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        return output.replace('\r\n', '\n').replace('\r', '\n').strip() == expected.replace('\r\n', '\n').replace('\r', '\n').strip()
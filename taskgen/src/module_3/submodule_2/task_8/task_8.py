from typing import Optional
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task8(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        rem = self.seed % 2
        step = max(1, self.seed % 10)
        if rem == 0:
            return {
                "func": "modify_int_array", "type": "int", "step": step,
                "add_val": self.seed % 1000, "is_not": False, "fmt": "%d"
            }
        else:
            return {
                "func": "modify_float_array", "type": "float", "step": step,
                "add_val": (self.seed % 1000) * 0.5, "is_not": True, "fmt": "%.1f"
            }

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>

void {p["func"]}({p["type"]} *arr);

int main() {{
    {p["type"]} arr[10] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
    {p["func"]}(arr);
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        cond_desc = "не кратной" if p["is_not"] else "кратной"
        return f"""### Тема: Арифметика указателей
**Сложность:** средняя

**Задание:**
Напишите функцию `void {p["func"]}({p["type"]} *arr)`, которая принимает указатель на массив из 10 элементов типа `{p["type"]}`.
Увеличьте каждый элемент, находящийся на позиции, {cond_desc} `{p["step"]}`, на значение `{p["add_val"]}`.
Используйте только арифметику указателей для доступа к элементам. Прямая индексация `arr[i]` запрещена.
Выведите итоговый массив в формате: `Array after modification: val1, val2, ..., val10`

**Формат вывода:** Array after modification: v0, v1, ..., v9
"""

    def _generate_tests(self):
        p = self._get_params()
        arr = list(range(10))
        step = p["step"]
        add = p["add_val"]
        is_not = p["is_not"]

        for i in range(10):
            if (i % step != 0) if is_not else (i % step == 0):
                arr[i] += add

        vals = ", ".join(p["fmt"] % v for v in arr)
        expected = f"Array after modification: {vals}\n"

        self.tests = [TestItem(
            input_str="", showed_input="", expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*{p["type"]}\s*\*\s*arr\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура функции `{p['func']}`."

        if re.search(r'\barr\s*\[', code):
            return "Ошибка: прямая индексация arr[i] запрещена. Используйте указательную арифметику."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
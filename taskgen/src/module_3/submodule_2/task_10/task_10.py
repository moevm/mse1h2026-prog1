from typing import Optional
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task10(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        v = self.seed % 2
        if v == 0:
            return {
                "func": "swap_double_ptrs",
                "fmt_str": "After swap: *a=%d, *b=%d\n",
                "check_fmt": r'After\s+swap:\s*\*a=%d,\s*\*b=%d'
            }
        else:
            return {
                "func": "exchange_ptr_targets",
                "fmt_str": "Swapped => a:%d b:%d\n",
                "check_fmt": r'Swapped\s*=>\s*a:%d\s*b:%d'
            }

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>

void {p["func"]}(int **a, int **b);

int main() {{
    int x1 = 10, y1 = 20;
    int *p1 = &x1, *p2 = &y1;
    {p["func"]}(&p1, &p2);

    int x2 = 5, y2 = 99;
    int *p3 = &x2, *p4 = &y2;
    {p["func"]}(&p3, &p4);

    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        return f"""### Тема: Указатель на указатель
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {p["func"]}(int **a, int **b)`, которая меняет местами два указателя через двойные указатели (обменивает адреса, а не значения).
После обмена выведите значения, на которые теперь указывают `*a` и `*b`, соблюдая формат `{p["fmt_str"].strip()}`. Используйте временный указатель `int *temp`. Прямое присваивание `**a = **b` запрещено.

**Формат вывода:** `{p["fmt_str"].strip()}`
"""

    def _generate_tests(self):
        p = self._get_params()
        if "After swap" in p["fmt_str"]:
            expected = f"After swap: *a=20, *b=10\nAfter swap: *a=99, *b=5\n"
        else:
            expected = f"Swapped => a:20 b:10\nSwapped => a:99 b:5\n"

        self.tests = [TestItem(
            input_str="",
            showed_input="double_ptr_swap_2_cases",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*int\s*\*\*\s*a\s*,\s*int\s*\*\*\s*b\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}(int **a, int **b)`."

        if re.search(r'\*\*a\s*=\s*\*\*b|\*\*b\s*=\s*\*\*a', code):
            return "Ошибка: обмен значений (**a = **b) запрещён. Меняйте адреса через указатели."

        temp_decl = r'int\s*\*\s*temp\s*=\s*\*a'
        swap_logic = r'\*a\s*=\s*\*b\s*;?\s*\*b\s*=\s*temp'
        if not re.search(temp_decl, code) or not re.search(swap_logic, code, re.DOTALL):
            return "Ошибка: обмен должен выполняться через `int *temp = *a; *a = *b; *b = temp;`."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        norm = lambda s: s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return norm(output) == norm(expected)
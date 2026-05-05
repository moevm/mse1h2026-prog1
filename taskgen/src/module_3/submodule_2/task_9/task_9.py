from typing import Optional
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule2_Task9(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_params(self) -> dict:
        v = self.seed % 2
        if v == 0:
            return {
                "func": "compare_ptrs",
                "msg_eq": "Same address",
                "has_prefix": False
            }
        else:
            return {
                "func": "ptr_equality_check",
                "msg_eq": "Equal pointers",
                "has_prefix": True
            }

    def _get_test_driver_code(self) -> str:
        p = self._get_params()
        return f'''#include <stdio.h>

void {p["func"]}(int *a, int *b);

int main() {{
    int x = 10, y = 20, z = 10;
    {p["func"]}(&x, &x);
    {p["func"]}(&x, &y);
    {p["func"]}(&y, &x);
    {p["func"]}(&x, &z);
    
    return 0;
}}
'''

    def generate_task(self) -> str:
        p = self._get_params()
        fmt_desc = "`*a` <op> `*b`" if not p["has_prefix"] else "Result: `*a` <op> `*b`"
        return f"""### Тема: Сравнение указателей
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {p["func"]}(int *a, int *b)`, которая проверяет, указывают ли указатели на один адрес.
- Если адреса совпадают (`a == b`), выведите `{p["msg_eq"]}`.
- Если адреса разные, сравните значения `*a` и `*b` и выведите результат в формате `{fmt_desc}`, подставив оператор (`<`, `>` или `==`).

**Формат вывода:**
- При равенстве адресов: `{p["msg_eq"]}`
- При разных адресах: `{fmt_desc.replace("<op>", "<оператор>")}`
"""

    def _generate_tests(self):
        p = self._get_params()
        if not p["has_prefix"]:
            expected = (
                f"{p['msg_eq']}\n"
                f"10 < 20\n"
                f"20 > 10\n"
                f"10 == 10\n"
            )
        else:
            expected = (
                f"{p['msg_eq']}\n"
                f"Result: 10 < 20\n"
                f"Result: 20 > 10\n"
                f"Result: 10 == 10\n"
            )

        self.tests = [TestItem(
            input_str="",
            showed_input="4_pointer_scenarios",
            expected=expected,
            compare_func=self._compare_default
        )]

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        p = self._get_params()
        code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', self.solution)

        sig = rf'void\s+{p["func"]}\s*\(\s*int\s*\*\s*a\s*,\s*int\s*\*\s*b\s*\)'
        if not re.search(sig, code):
            return f"Ошибка: неверная сигнатура `{p['func']}(int *a, int *b)`."

        if not re.search(r'\ba\s*==\s*b\b|\bb\s*==\s*a\b', code):
            return "Ошибка: необходимо проверить равенство адресов указателей (a == b)."

        val_cmp = r'\*a\s*(<|>|==|!=)\s*\*b|\*b\s*(<|>|==|!=)\s*\*a'
        if not re.search(val_cmp, code):
            return "Ошибка: необходимо сравнить значения по указателям (*a и *b)."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        norm = lambda s: s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return norm(output) == norm(expected)
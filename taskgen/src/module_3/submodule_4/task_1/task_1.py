from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task1(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_test_driver_code(self) -> str:
        names = ["my_strlen", "calc_str_len", "get_string_size", "ptr_strlen"]
        func_name = names[self.seed % 4]
        return f'''#include <stdio.h>

int {func_name}(char *s);

int main() {{
    char s[256];
    if (scanf("%255s", s) != 1) return 1;
    printf("%d\\n", {func_name}(s));
    return 0;
}}
'''

    def generate_task(self) -> str:
        names = ["my_strlen", "calc_str_len", "get_string_size", "ptr_strlen"]
        func_name = names[self.seed % 4]
        return f"""### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `int {func_name}(char *s)`, которая возвращает количество символов в строке (без учёта `'\\0'`).
```"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        base_strings = ["a", "hello", "test123", "C", "programming", "x", "abc", "C_Programming", "data"]

        while len(base_strings) < self.tests_num:
            base_strings.append("".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789_", k=random.randint(1, 20))))

        for s in base_strings[:self.tests_num]:
            input_str = s
            expected = f"{len(s)}\n"
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f's = "{s}"',
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        names = ["my_strlen", "calc_str_len", "get_string_size", "ptr_strlen"]
        func_name = names[self.seed % 4]

        sig_pattern = rf'int\s+{func_name}\s*\(\s*char\s*\*\s*s\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if '#include <string.h>' in self.solution or '#include "string.h"' in self.solution:
            return "Ошибка: запрещено подключать <string.h>."
        if re.search(r'\bstrlen\b', self.solution):
            return "Ошибка: запрещено использовать библиотечную функцию strlen."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
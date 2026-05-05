from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task3(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_test_driver_code(self) -> str:
        names = ["my_strcat", "custom_strcat", "append_string", "manual_strcat"]
        func_name = names[self.seed % 4]
        return f'''#include <stdio.h>

char *{func_name}(char *dest, char *src);

int main() {{
    char dest[512] = "base_"; // Фиксированный префикс для проверки именно ДОБАВЛЕНИЯ
    char src[256];
    if (scanf("%255s", src) != 1) return 1;
    
    char *res = {func_name}(dest, src);
    printf("%s\\n", res);
    return 0;
}}
'''

    def generate_task(self) -> str:
        names = ["my_strcat", "custom_strcat", "append_string", "manual_strcat"]
        func_name = names[self.seed % 4]
        return f"""### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *dest, char *src)`, которая приписывает строку `src` к концу `dest`.
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        base_strings = [
            "hello", "C_Programming", "test123", "data", 
            "x", "abc", "concat", "buffer", "string_op"
        ]
        while len(base_strings) < self.tests_num:
            base_strings.append("".join(
                random.choices("abcdefghijklmnopqrstuvwxyz0123456789_", k=random.randint(3, 12))
            ))

        for s in base_strings[:self.tests_num]:
            input_str = s
            expected = f"base_{s}\n"
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f'dest="base_", src="{s}"',
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        names = ["my_strcat", "custom_strcat", "append_string", "manual_strcat"]
        func_name = names[self.seed % 4]

        sig_pattern = rf'char\s*\*\s*{func_name}\s*\(\s*char\s*\*\s*dest\s*,\s*char\s*\*\s*src\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if '#include <string.h>' in self.solution or '#include "string.h"' in self.solution:
            return "Ошибка: запрещено подключать <string.h>."
        if re.search(r'\bstrcat\b', self.solution):
            return "Ошибка: запрещено использовать библиотечную функцию strcat."
        if 'return' not in self.solution:
            return "Ошибка: функция должна возвращать указатель на начало dest."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines).strip()
        return normalize(output) == normalize(expected)
from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task4(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_test_driver_code(self) -> str:
        names = ["my_strcmp", "custom_strcmp", "str_compare", "manual_strcmp"]
        func_name = names[self.seed % 4]
        return f'''#include <stdio.h>

int {func_name}(char *s1, char *s2);

int main() {{
    char s1[256], s2[256];
    if (scanf("%255s %255s", s1, s2) != 2) return 1;
    printf("%d\\n", {func_name}(s1, s2));
    return 0;
}}
'''

    def generate_task(self) -> str:
        names = ["my_strcmp", "custom_strcmp", "str_compare", "manual_strcmp"]
        func_name = names[self.seed % 4]
        return f"""### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `int {func_name}(char *s1, char *s2)`, которая сравнивает две строки.
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        base_pairs = [
            ("hello", "hello"), ("abc", "abd"), ("xyz", "xya"),
            ("test", "testing"), ("C", "c"), ("data", "date"),
            ("hello", "hellp"), ("a", "b"), ("long_string", "long_strinG")
        ]

        while len(base_pairs) < self.tests_num:
            len1 = random.randint(1, 12)
            len2 = random.randint(1, 12)
            s1 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=len1))
            s2 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=len2))
            base_pairs.append((s1, s2))

        for s1, s2 in base_pairs[:self.tests_num]:
            input_str = f"{s1} {s2}"

            expected_val = 0
            for c1, c2 in zip(s1, s2):
                if c1 != c2:
                    expected_val = ord(c1) - ord(c2)
                    break
            else:
                if len(s1) != len(s2):
                    if len(s1) > len(s2):
                        expected_val = ord(s1[len(s2)]) 
                    else:
                        expected_val = -ord(s2[len(s1)])

            expected = f"{expected_val}\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f's1="{s1}", s2="{s2}"',
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        names = ["my_strcmp", "custom_strcmp", "str_compare", "manual_strcmp"]
        func_name = names[self.seed % 4]

        sig_pattern = rf'int\s+{func_name}\s*\(\s*char\s*\*\s*s1\s*,\s*char\s*\*\s*s2\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if '#include <string.h>' in self.solution or '#include "string.h"' in self.solution:
            return "Ошибка: запрещено подключать <string.h>."
        if re.search(r'\bstrcmp\b', self.solution):
            return "Ошибка: запрещено использовать библиотечную функцию strcmp."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            return s.strip()
        return normalize(output) == normalize(expected)
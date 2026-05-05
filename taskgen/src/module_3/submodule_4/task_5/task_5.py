from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem, DEFAULT_TEST_NUM


class Module3_Submodule4_Task5(BaseTaskClass):
    def __init__(self, **kwargs):
        default_params = {"tests_num": DEFAULT_TEST_NUM}
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.check_files = {"test_driver.c": self._get_test_driver_code()}

    def _get_test_driver_code(self) -> str:
        names = ["my_strchr", "custom_strchr", "find_first_char", "manual_strchr"]
        func_name = names[self.seed % 4]
        return f'''#include <stdio.h>

char *{func_name}(char *s, char c);

int main() {{
    char s[256];
    char c;
    // Пробел перед %c обязателен: он пропускает перевод строки после слова
    if (scanf("%255s %c", s, &c) != 2) return 1;
    
    char *res = {func_name}(s, c);
    if (res != NULL) printf("Found: %s\\n", res);
    else printf("Not found\\n");
    
    return 0;
}}
'''

    def generate_task(self) -> str:
        names = ["my_strchr", "custom_strchr", "find_first_char", "manual_strchr"]
        func_name = names[self.seed % 4]
        return f"""### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *s, char c)`, которая находит первое вхождение символа `c` в строке `s`. 
"""

    def _generate_tests(self):
        random.seed(self.seed)
        self.tests = []

        base_pairs = [
            ("hello", 'l'), ("test", 'z'), ("abc", 'a'),
            ("data", 'a'), ("unique", 'q'), ("programming", 'r'),
            ("string", 'g'), ("find", 'n'), ("code", 'c'), ("void", 'v')
        ]

        while len(base_pairs) < self.tests_num:
            s_len = random.randint(3, 12)
            s = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=s_len))
            if random.random() < 0.7 and len(s) > 0:
                c = random.choice(s)
            else:
                c = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*")
            base_pairs.append((s, c))

        for s, c in base_pairs[:self.tests_num]:
            input_str = f"{s} {c}"

            idx = s.find(c)
            if idx != -1:
                expected = f"Found: {s[idx:]}\n"
            else:
                expected = "Not found\n"

            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=f's="{s}", c="{c}"',
                expected=expected,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        names = ["my_strchr", "custom_strchr", "find_first_char", "manual_strchr"]
        func_name = names[self.seed % 4]

        sig_pattern = rf'char\s*\*\s*{func_name}\s*\(\s*char\s*\*\s*s\s*,\s*char\s+c\s*\)'
        if not re.search(sig_pattern, self.solution):
            return f"Ошибка: функция {func_name} имеет неверную сигнатуру или отсутствует."

        if '#include <string.h>' in self.solution or re.search(r'\bstrchr\b', self.solution):
            return "Ошибка: запрещено использовать <string.h> или библиотечную strchr."

        return None

    def _compare_default(self, output: str, expected: str) -> bool:
        def normalize(s: str) -> str:
            return s.replace('\r\n', '\n').replace('\r', '\n').strip()
        return normalize(output) == normalize(expected)
from typing import Optional
import random
from pathlib import Path
import sys
import re

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from base_module.base_task import *


class Task2_1_1_DeclarePointer(BaseTaskClass):
    """2.1.1: Объявите указатель с именем ptr на тип int"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""

    def generate_task(self) -> str:
        return """Объявите указатель с именем ptr на тип int."""

    def _generate_tests(self):
        self.tests = []

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if not re.search(r'\bint\s*\*\s*ptr\b', self.solution):
            return "Ошибка: Не найдено объявление."

        if re.search(r'\bint\s*\*\s*ptr\s*=', self.solution):
            return "Предупреждение: В этой задаче требуется только объявление, без инициализации."
        
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        return True, "OK"

class Task2_2_1_InitPointer(BaseTaskClass):
    """2.2.1: Объявите указатель с именем ptr на тип int и инициализируйте адресом переменной a."""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""

    def generate_task(self) -> str:
        return """Объявите указатель с именем ptr на тип int и инициализируйте адресом переменной a."""

    def _generate_tests(self):
        self.tests = []

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if not re.search(r'\bint\s*\*\s*ptr\s*=\s*&a\s*;', self.solution):
            return "Ошибка: Указатель ptr должен быть инициализирован адресом переменной a."
        
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        return True, "OK"

class Task2_3_1_NullCheckValidator(BaseTaskClass):
    """2.3.1: Проверка указателя на NULL"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""
        
        self.static_check_files = {
            "main.c": self._get_main_wrapper()
        }

    def _get_main_wrapper(self) -> str:
        return r"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

void solution(int *ptr);

int main() {
    int value = 42;
    int *ptr = &value;
    
    int test_mode = 0;
    if (scanf("%d", &test_mode) == 1 && test_mode == 1) {
        ptr = NULL;  
    }
    
    solution(ptr);
    return 0;
}
"""

    def generate_task(self) -> str:
        return """
Напишите функцию void solution(int *ptr), которая:
1. Проверит указатель ptr на NULL
2. Если ptr != NULL, выведет значение: "Value: X"
3. Если ptr == NULL, выведет: "NULL pointer"
"""

    def _generate_tests(self):
        self.tests = []

        self.tests.append(TestItem(
            input_str="0",
            showed_input="ptr = &value",
            expected="Value:",
            compare_func=self._compare_value_test
        ))

        self.tests.append(TestItem(
            input_str="1",
            showed_input="ptr = NULL",
            expected="NULL pointer",
            compare_func=self._compare_null_test
        ))

    def _compare_value_test(self, output: str, expected: str) -> bool:
        return "Value:" in output and "NULL" not in output

    def _compare_null_test(self, output: str, expected: str) -> bool:
        return "NULL pointer" in output

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if "NULL" not in self.solution and "== 0" not in self.solution and "== NULL" not in self.solution:
            return "Ошибка: Не обнаружена проверка указателя на NULL."

        if "if" not in self.solution:
            return "Ошибка: Не обнаружена условная проверка (if)."

        if "*" not in self.solution:
            return "Ошибка: Не обнаружено разыменование указателя."
        
        return None

    def compile(self) -> Optional[str]:
        import subprocess
        
        self._dump_files(self.static_check_files)
        
        with open("solution.c", "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")
        
        p = subprocess.run(
            ["gcc", "-c", "main.c", "-o", "main.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции main.c:\n{p.stdout}"
        
        p = subprocess.run(
            ["gcc", "-c", "solution.c", "-o", "stud_work.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции solution.c:\n{p.stdout}"
        
        link_args = ["-O2", "-Wall", "-mconsole"]
        p = subprocess.run(
            ["gcc", "main.o", "stud_work.o", "-o", "prog.x"] + link_args,
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout}"
        
        return None

class Task2_4_1_PointerBasicsValidator(BaseTaskClass):
    "2.4.1: Присваивание значения value по адресу ptr."
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""
        
        self.static_check_files = {
            "main.c": self._get_main_wrapper()
        }

    def _get_main_wrapper(self) -> str:
        return r"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

void solution(int *ptr, int value);

int main() {
    int var = 0;
    int *ptr = &var;
    
    if (scanf("%d", &var) != 1) {
        return 1;
    }
    
    solution(ptr, var);
    
    printf("%d", var);
    
    return 0;
}
"""

    def generate_task(self) -> str:
        return """
Разыменовывание указателей.
Напишите функцию void solution(int *ptr, int value), которая запишет 
значение value по адресу ptr.
"""

    def _generate_tests(self):
        self.tests = []
        test_values = [0, 1, -1, 100, -100, 42, 999]
        
        for val in test_values:
            self.tests.append(TestItem(
                input_str=str(val),
                showed_input=val,
                expected=str(val),
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if "*" not in self.solution:
            return "Ошибка: Не обнаружено использование указателей (символ *)."
        
        import re
        if not re.search(r'\*\s*ptr\s*=', self.solution):
            return "Ошибка: Указатель объявлен, но не используется для записи значения (нет разыменования *ptr)."
        
        return None

    def compile(self) -> Optional[str]:
        import subprocess
        import os

        self._dump_files(self.static_check_files)

        with open("solution.c", "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")

        p = subprocess.run(
            ["gcc", "-c", "main.c", "-o", "main.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции main.c:\n{p.stdout}"

        p = subprocess.run(
            ["gcc", "-c", "solution.c", "-o", "stud_work.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции solution.c:\n{p.stdout}"
        
        link_args = ["-O2", "-Wall", "-mconsole"]
        p = subprocess.run(
            ["gcc", "main.o", "stud_work.o", "-o", "prog.x"] + link_args,
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout}"
        
        return None

class Task2_5_1_PrintAddressValidator(BaseTaskClass):
    """2.5.1: Вывод адреса переменной через указатель"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""

    def generate_task(self) -> str:
        return """
Вывод адреса переменной.
Объявите целочисленную переменную a равную 512, передайте адрес переменной указателю ptr.
Выведите на экран: "Переменная a = 512 хранится по адресу 0x..."
"""

    def _generate_tests(self):
        self.tests = []

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if not re.search(r'\bint\s+a\s*=\s*512\s*;', self.solution):
            return "Ошибка: Не найдено объявление 'int a = 512;'."

        if not re.search(r'\bint\s*\*\s*ptr\s*=\s*&a\s*;', self.solution):
            return "Ошибка: Указатель ptr должен быть инициализирован адресом a."

        if '%p' not in self.solution:
            return "Ошибка: Не найден формат вывода адреса (%p)."

        if '(void*)' not in self.solution and '(void *)' not in self.solution:
            return "Предупреждение: Рекомендуется приведение указателя к (void*) для printf."
        
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        return True, "OK"

class Task2_5_2_DereferenceBeforeAfterValidator(BaseTaskClass):
    """2.5.2: Изменение значения через указатель с выводом до и после"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""
        
        self.static_check_files = {
            "main.c": self._get_main_wrapper()
        }

    def _get_main_wrapper(self) -> str:
        return r"""
#include <stdio.h>
#include <stdlib.h>

void solution(int *ptr, int new_value);

int main() {
    int a = 10;
    int *ptr = &a;
    int new_value = 20;
    
    // Вывод до изменения
    printf("Before: a = %d\n", a);
    
    solution(ptr, new_value);
    
    // Вывод после изменения
    printf("After: a = %d\n", a);
    
    return 0;
}
"""

    def generate_task(self) -> str:
        return """
Изменение значения через указатель.

Напишите функцию void solution(int *ptr, int new_value), которая:
1. Выведет значение *ptr до изменения
2. Изменит значение по адресу ptr на new_value
3. Выведет значение *ptr после изменения

Формат вывода:
Before: X
After: Y
"""

    def _generate_tests(self):
        self.tests = []
        # Тест с new_value = 20
        self.tests.append(TestItem(
            input_str="",
            showed_input="a=10, new_value=20",
            expected="Before: 10\nAfter: 20",
            compare_func=self._compare_output
        ))

    def _compare_output(self, output: str, expected: str) -> bool:
        return "Before: 10" in output and "After: 20" in output

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if '*ptr' not in self.solution:
            return "Ошибка: Не обнаружено разыменование указателя (*ptr)."

        if not re.search(r'\*\s*ptr\s*=', self.solution):
            return "Ошибка: Не найдено присваивание через указатель (*ptr = ...)."

        if self.solution.count('printf') < 2:
            return "Ошибка: Требуется минимум два вывода (до и после изменения)."
        
        return None

    def compile(self) -> Optional[str]:
        import subprocess
        
        self._dump_files(self.static_check_files)
        
        with open("solution.c", "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")
        
        p = subprocess.run(
            ["gcc", "-c", "main.c", "-o", "main.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции main.c:\n{p.stdout}"
        
        p = subprocess.run(
            ["gcc", "-c", "solution.c", "-o", "stud_work.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции solution.c:\n{p.stdout}"
        
        link_args = ["-O2", "-Wall", "-mconsole"]
        p = subprocess.run(
            ["gcc", "main.o", "stud_work.o", "-o", "prog.x"] + link_args,
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout}"
        
        return None

class Task2_6_1_PointerIncDecValidator(BaseTaskClass):
    """2.6.1: Инкремент и декремент указателя"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""

    def generate_task(self) -> str:
        return """
Арифметика указателей (++, --).

Объявите переменную int n = 10;
Объявите указатель int *ptr = &n;

Выведите адрес и значение ptr:
1. До изменения
2. После ptr++
3. После ptr--

Формат вывода для каждого шага:
address=0x... value=X
"""

    def _generate_tests(self):
        self.tests = []

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if not re.search(r'\bint\s+n\s*=\s*10\s*;', self.solution):
            return "Ошибка: Не найдено объявление 'int n = 10;'."

        if not re.search(r'\bint\s*\*\s*ptr\s*=\s*&n\s*;', self.solution):
            return "Ошибка: Указатель ptr должен быть инициализирован адресом n."

        if '++' not in self.solution or '--' not in self.solution:
            return "Ошибка: Требуется использование ptr++ и ptr--."

        if '%p' not in self.solution:
            return "Ошибка: Не найден формат вывода адреса (%p)."

        if self.solution.count('printf') < 3:
            return "Ошибка: Требуется минимум три вывода (до, после ++, после --)."
        
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        return True, "OK"

class Task2_6_2_PointerArithmeticValidator(BaseTaskClass):
    """2.6.2: Увеличение каждого элемента массива на 10, используя арифметику указателей."""

    def __init__(self, *args, array_length: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.array_length = array_length
        self.jail_path = ""
        self.jail_exec = ""
        
        self.static_check_files = {
            "main.c": self._get_main_wrapper()
        }

    def _get_main_wrapper(self) -> str:
        return f"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

void solution(int *arr, int len);

int main() {{
    int len = {self.array_length};
    int *arr = (int*)calloc(len, sizeof(int));
    
    for(int i = 0; i < len; i++) {{
        if(scanf("%d", &arr[i]) != 1) {{
            return 1;
        }}
    }}
    
    solution(arr, len);
    
    for(int i = 0; i < len; i++) {{
        printf("%d ", arr[i]);
    }}
    
    free(arr);
    return 0;
}}
"""

    def generate_task(self) -> str:
        return f"""
Арифметика указателей.
Дан массив из {self.array_length} элементов.
Требуется увеличить каждый элемент на 10, используя ТОЛЬКО арифметику указателей.

Требования:
1. Запрещено использование оператора []
2. Используйте разыменование и арифметику указателей (*ptr, ptr++)

void solution(int *arr, int len) {{
    // Ваш код здесь
}}
"""

    def _generate_tests(self):
        self.tests = []
        for _ in range(5):
            arr = [random.randint(-100, 100) for _ in range(self.array_length)]
            input_str = " ".join(map(str, arr))
            expected_arr = [x + 10 for x in arr]
            expected_str = " ".join(map(str, expected_arr))
            
            self.tests.append(TestItem(
                input_str=input_str,
                showed_input=arr,
                expected=expected_str,
                compare_func=self._compare_default
            ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if "[" in self.solution:
            return "Ошибка: Использование оператора [] запрещено. Используйте указатели."

        if "*" not in self.solution:
            return "Ошибка: Не обнаружено разыменование указателя."
        
        if "++" not in self.solution and "--" not in self.solution and "+" not in self.solution:
            return "Ошибка: Не обнаружена арифметика указателей (++, --, или +)."
        
        return None

    def compile(self) -> Optional[str]:
        import subprocess
        
        self._dump_files(self.static_check_files)
        
        with open("solution.c", "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")

        p = subprocess.run(
            ["gcc", "-c", "main.c", "-o", "main.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции main.c:\n{p.stdout}"
        
        p = subprocess.run(
            ["gcc", "-c", "solution.c", "-o", "stud_work.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции solution.c:\n{p.stdout}"

        link_args = ["-O2", "-Wall", "-mconsole"]
        p = subprocess.run(
            ["gcc", "main.o", "stud_work.o", "-o", "prog.x"] + link_args,
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout}"
        
        return None

class Task2_7_1_PointerComparisonValidator(BaseTaskClass):
    """
    Валидатор для сравнения указателей.
    ptr1 == ptr2, ptr1 < ptr2, ptr1 > ptr2
    """

    def __init__(self, *args, array_length: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.array_length = array_length
        self.jail_path = ""
        self.jail_exec = ""
        
        self.static_check_files = {
            "main.c": self._get_main_wrapper()
        }

    def _get_main_wrapper(self) -> str:
        return f"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

int solution(int *arr, int len);

int main() {{
    int len = {self.array_length};
    int *arr = (int*)calloc(len, sizeof(int));
    
    for(int i = 0; i < len; i++) {{
        arr[i] = i * 10;
    }}
    
    int result = solution(arr, len);
    printf("%d", result);
    
    free(arr);
    return 0;
}}
"""

    def generate_task(self) -> str:
        return f"""
Сравнение указателей.

Дан массив из {self.array_length} элементов.
Напишите функцию int solution(int *arr, int len), которая:
1. Объявит два указателя: один на начало массива, другой на конец
2. Сравнит их (ptr_start < ptr_end)
3. Вернёт 1, если ptr_start < ptr_end, иначе 0

Ожидаемый результат: 1
"""

    def _generate_tests(self):
        self.tests = []
        self.tests.append(TestItem(
            input_str="",
            showed_input="arr[5]",
            expected="1",
            compare_func=self._compare_default
        ))

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if "<" not in self.solution and ">" not in self.solution and "==" not in self.solution:
            return "Ошибка: Не обнаружено сравнение указателей (<, >, ==)."

        if "ptr" not in self.solution.lower() and "p1" not in self.solution.lower() and "p2" not in self.solution.lower():
            return "Ошибка: Не обнаружены указатели для сравнения."

        if "return" not in self.solution:
            return "Ошибка: Функция должна возвращать результат."
        
        return None

    def compile(self) -> Optional[str]:
        import subprocess
        
        self._dump_files(self.static_check_files)
        
        with open("solution.c", "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")
        
        p = subprocess.run(
            ["gcc", "-c", "main.c", "-o", "main.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции main.c:\n{p.stdout}"
        
        p = subprocess.run(
            ["gcc", "-c", "solution.c", "-o", "stud_work.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции solution.c:\n{p.stdout}"
        
        link_args = ["-O2", "-Wall", "-mconsole"]
        p = subprocess.run(
            ["gcc", "main.o", "stud_work.o", "-o", "prog.x"] + link_args,
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout}"
        
        return None


class Task2_8_1_PointerDifferenceValidator(BaseTaskClass):
    """2.8.1: Разность указателей"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""

    def generate_task(self) -> str:
        return """
Разность указателей.

Объявите целочисленную переменную val равную 42, передайте ее адрес указателю p1,
далее передайте адрес указателя p1 указателю p2.

Выведите разность адресов p2 и p1

Требования:
- Используйте арифметику указателей (p2 - p1)
- Выведите результат через printf("%ld", ...) или printf("%d", ...)
"""

    def _generate_tests(self):
        self.tests = []

    def check_sol_prereq(self) -> Optional[str]:
        err = super().check_sol_prereq()
        if err:
            return err

        if not re.search(r'\bint\s+val\s*=\s*42\s*;', self.solution):
            return "Ошибка: Не найдено объявление 'int val = 42;'."

        if not re.search(r'\bint\s*\*\s*p1\s*=\s*&val\s*;', self.solution):
            return "Ошибка: Указатель p1 должен быть инициализирован адресом val."

        if not re.search(r'\bint\s*\*\*\s*p2\s*=\s*&p1\s*;', self.solution):
            return "Ошибка: Указатель p2 должен быть инициализирован адресом p1."

        if 'p2 - p1' not in self.solution and 'p2-p1' not in self.solution:
            return "Ошибка: Не найдена операция вычитания указателей (p2 - p1)."

        if 'printf' not in self.solution:
            return "Ошибка: Требуется вывод результата через printf."
        
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        return True, "OK"

class Task2_9_1_TriplePointerValidator(BaseTaskClass):
    """
    Валидатор для указателя на указатель на указатель (***ptr).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jail_path = ""
        self.jail_exec = ""
        
        self.static_check_files = {
            "main.c": self._get_main_wrapper()
        }

    def _get_main_wrapper(self) -> str:
        return r"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

void solution(int ***ptr);

int main() {
    int value = 50;
    int *p1 = &value;
    int **p2 = &p1;
    int ***p3 = &p2;
    
    solution(p3);
    
    printf("%d", value);
    return 0;
}
"""

    def generate_task(self) -> str:
        return """
Указатель на указатель на указатель (***ptr).

Дано: int ***ptr (тройной указатель).
Исходное значение: 50

Напишите функцию void solution(int ***ptr), которая:
1. Изменит исходное значение на 100, используя только ptr
2. Для доступа к значению используйте ***ptr (3 звезды)

Ожидаемый результат: 100
"""

    def _generate_tests(self):
        self.tests = []
        self.tests.append(TestItem(
            input_str="",
            showed_input="value = 50",
            expected="100",
            compare_func=self._compare_default
        ))

    def check_sol_prereq(self) -> Optional[str]:
        import re
        err = super().check_sol_prereq()
        if err:
            return err

        if not re.search(r'\*\*\*ptr\s*=', self.solution):
            return "Ошибка: Не обнаружено разыменование ***ptr = (3 звезды в присваивании)."

        if "=" not in self.solution:
            return "Ошибка: Не обнаружено присваивание значения."

        if "value" in self.solution.lower():
            return "Ошибка: Прямой доступ к переменной value запрещён."
        
        return None

    def compile(self) -> Optional[str]:
        import subprocess
        
        self._dump_files(self.static_check_files)
        
        with open("solution.c", "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")
        
        p = subprocess.run(
            ["gcc", "-c", "main.c", "-o", "main.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции main.c:\n{p.stdout}"
        
        p = subprocess.run(
            ["gcc", "-c", "solution.c", "-o", "stud_work.o", "-O2", "-Wall"],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка компиляции solution.c:\n{p.stdout}"
        
        link_args = ["-O2", "-Wall", "-mconsole"]
        p = subprocess.run(
            ["gcc", "main.o", "stud_work.o", "-o", "prog.x"] + link_args,
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout}"
        
        return None
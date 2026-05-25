from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_literal(seed: int) -> str:
    random.seed(seed)
    choice = random.choice(['int', 'uint', 'long', 'ulong', 'longlong', 'ulonglong', 'hex', 'oct', 'float', 'double'])
    if choice == 'int':
        value = random.randint(-2147483648, 2147483647)
        return str(value)
    elif choice == 'uint':
        value = random.randint(0, 4294967295)
        return f"{value}U"
    elif choice == 'long':
        value = random.randint(-9223372036854775808, 9223372036854775807)
        return f"{value}L"
    elif choice == 'ulong':
        value = random.randint(0, 18446744073709551615)
        return f"{value}UL"
    elif choice == 'longlong':
        value = random.randint(-9223372036854775808, 9223372036854775807)
        return f"{value}LL"
    elif choice == 'ulonglong':
        value = random.randint(0, 18446744073709551615)
        return f"{value}ULL"
    elif choice == 'hex':
        value = random.randint(0, 0xFFFFFFFF)
        return f"0x{value:X}"
    elif choice == 'oct':
        value = random.randint(0, 0o777777777)
        return f"0{value:o}"
    elif choice == 'float':
        value = random.uniform(-1e6, 1e6)
        return f"{value}f"
    else:
        value = random.uniform(-1e6, 1e6)
        return str(value)


def determine_type(literal: str) -> str:
    lit = literal.strip()
    if lit.endswith('f') or lit.endswith('F'):
        return "float"
    if 'e' in lit or 'E' in lit or '.' in lit:
        if lit.endswith(('l', 'L')):
            return "long double"
        return "double"
    if lit.endswith(('ULL', 'Ull', 'uLL', 'ull')):
        return "unsigned long long"
    if lit.endswith(('LL', 'Ll', 'lL')):
        return "long long"
    if lit.endswith(('UL', 'Ul', 'uL', 'ul')):
        return "unsigned long"
    if lit.endswith(('L', 'l')):
        return "long"
    if lit.endswith(('U', 'u')):
        return "unsigned int"
    if lit.startswith(('0x', '0X')):
        return "int"
    if lit.startswith('0') and len(lit) > 1:
        return "int"
    if int(lit) < 0:
        return "int"
    else:
        return "int"


class Module_1_Submodule_3_task_5(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.literal = generate_literal(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return f"Определите тип литерала `{self.literal}`. Запишите название типа (например, int, unsigned int и т.д.)."

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = determine_type(self.literal)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda out, exp: out.strip().lower() == exp.strip().lower()
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.student_solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self):
        try:
            self.generate_task()
            expected = determine_type(self.literal)
            student = self.student_solution.strip()
            if student.lower() == expected.lower():
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
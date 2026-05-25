from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem
import struct


def generate_value(seed: int) -> float:
    n = seed % 2001 - 1000
    d = (seed // 2001) % 100 + 1
    return round(n / d, 6)


def float_to_hex(value: float) -> str:
    packed = struct.pack('f', value)
    int_val = int.from_bytes(packed, byteorder='big')
    return f"0x{int_val:08X}".lower()


def compare_answer(output: str, expected: str) -> bool:
    return output.strip().replace(" ", "").lower() == expected.strip().replace(" ", "").lower()


class Module_1_Submodule_2_task_13(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = generate_value(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"Представьте вещественное число `{self.value}` в формате IEEE 754 для типа `float` и "
            "запишите его шестнадцатеричное представление в виде `0xHHHHHHHH` "
            "(8 шестнадцатеричных цифр). Используйте строчные буквы. "
            "Сначала представьте число в формате знак, порядок, мантисса. "
            "Затем переведите в шестнадцатеричный формат. Ответом является шестнадцатеричное представление."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = float_to_hex(self.value)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: compare_answer(output, exp)
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
            expected = float_to_hex(self.value)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
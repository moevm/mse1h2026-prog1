from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem


class Module_1_Submodule_5_task_5(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operations = ["сумма", "разность", "произведение", "остаток от деления", "целая часть при делении"]
        self.operation = random.Random(self.seed).choice(self.operations)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (f"Есть две переменные, хранящие целые числа: const1 и const2, они уже инициализированы в программе. "
                f"Напишите строку кода для нахождения {self.operation}. Ответ должен быть записан в переменной answer, "
                f"которая раньше не встречалась.")

    def compile(self) -> Optional[str]:
        return None

    def _canonical_code(self) -> str:
        if self.operation == "сумма":
            return "int answer = const1 + const2;"
        elif self.operation == "разность":
            return "int answer = const1 - const2;"
        elif self.operation == "произведение":
            return "int answer = const1 * const2;"
        elif self.operation == "остаток от деления":
            return "int answer = const1 % const2;"
        else:
            return "int answer = const1 / const2;"

    def _is_correct(self, student: str, expected_op: str) -> bool:
        student_clean = re.sub(r'\s+', '', student)
        if not student_clean.startswith("intanswer=") or not student_clean.endswith(";"):
            return False
        right = student_clean[len("intanswer="):-1]
        if expected_op == "сумма":
            return right == "const1+const2" or right == "const2+const1"
        elif expected_op == "разность":
            return right == "const1-const2"
        elif expected_op == "произведение":
            return right == "const1*const2" or right == "const2*const1"
        elif expected_op == "остаток от деления":
            return right == "const1%const2"
        else:
            return right == "const1/const2"

    def _generate_tests(self):
        expected_code = self._canonical_code()
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected_code,
                compare_func=lambda out, exp: self._is_correct(out, self.operation)
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
            if self._is_correct(self.student_solution, self.operation):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_value(seed: int) -> int:
    return (seed % 201) - 100


def expected_line(value: int) -> str:
    return f"_Bool flag = {value} > 0; {'true' if value > 0 else 'false'}"


def compare_answer(output: str, expected: str) -> bool:
    return " ".join(output.strip().split()) == " ".join(expected.strip().split())


class Module_1_Submodule_2_task_9(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = generate_value(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        code = (
            "#include <stdio.h>\n\n"
            "int main()\n"
            "{\n"
            "    // пропущенная строка\n"
            "    if (flag == 1)\n"
            '        printf("true");\n'
            "    else\n"
            '        printf("false");\n'
            "    return 0;\n"
            "}"
        )
        return (
            "Есть программа с пропущенной строкой. Напишите пропущенную строку. "
            "В ней должна быть объявлена переменная `flag` типа `_Bool`, и присвоено ей значение `1`, "
            "если `value` положительное, и `0` в противном случае. (Сравнение должно быть прописано в программе, "
            "а не сразу задано `0` или `1`.) Через пробел напишите, что выведет дополненная программа. "
            f"Шаблон ответа: `пропущенная строкa; значение`.\n\n{code}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = expected_line(self.value)
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
            expected = expected_line(self.value)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
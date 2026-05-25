from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_bounds(seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    start = rng.randint(-30, 20)
    end = start + rng.randint(1, 20)
    return start, end


def choose_type(start: int, end: int) -> str:
    if start < 0 or end < 0:
        return "int"
    return "unsigned int"


def check_answer(start: int, end: int) -> str:
    integer_type = choose_type(start, end)
    return f"{integer_type} i = {start};"


def normalize_decl(text: str) -> str:
    return " ".join(
        text.replace(";", " ; ").replace("=", " = ").split()
    )


class Module_1_Submodule_2_task_3(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start, self.end = generate_bounds(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            "В программе есть цикл, перебирающий целые числа "
            f"от {self.start} до {self.end} включительно. "
            "Для счетчика цикла нужно объявить переменную i. "
            "Выберите наиболее подходящий целочисленный тип "
            "(int или unsigned int) для i. "
            "Напишите строку кода, объявляющую переменную i "
            "и присваивающую ей начальное значение start."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.start, self.end)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: normalize_decl(output) == normalize_decl(exp)
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
            expected = check_answer(self.start, self.end)
            student = self.student_solution.strip()
            if normalize_decl(student) == normalize_decl(expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
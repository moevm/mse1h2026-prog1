from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem

TASK_DESCRIPTION = "В каком сегменте памяти хранится {var}?"

VARIANTS = (
    ("локальная переменная внутри функции", "Stack"),
    ("динамически выделенная память", "Heap"),
    ("глобальная переменная с инициализацией", "Data"),
    ("глобальная переменная без инициализации", "BSS")
)

class Task1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.seed)
        self.correct = None
        self.stage = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        self.stage, self.correct = self.rng.choice(VARIANTS)
        return TASK_DESCRIPTION.format(var=self.stage)

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = self.correct
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(output.strip(), exp)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.student_solution.strip() 
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def _compare_default(self, output: str, expected: str) -> bool:
        return output.lower() == expected.lower()

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self):
        try:
            self.generate_task()
            
            if self.student_solution.strip().lower() == self.correct.lower():
                return True, "OK: Верный ответ."
            else:
                return False, f"FAIL: Ожидалось {self.correct}, получено {self.student_solution}"
        except Exception as e:
            return False, f"FAIL: {str(e)}"
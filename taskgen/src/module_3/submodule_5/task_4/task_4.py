from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem

TASK_DESCRIPTION = """
# Тема: Динамическая память

**Сложность:** легкая

**Задание:** Какая функция соответствует описанию: выделяет блок памяти размером size байт; возвращает указатель на начало блока или NULL при ошибке?
"""

class Module3_Submodule5_Task4(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.seed)
        self.correct = None
        self.stage = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        self.correct = "malloc"
        return TASK_DESCRIPTION

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
                return False, f"FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
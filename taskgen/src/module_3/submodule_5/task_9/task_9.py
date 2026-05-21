from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem

TASK_DESCRIPTION = """### Тема: Проверка на NULL и как это важно
**Сложность:** легкая

**Задание:**
В каком случае функция malloc(), calloc(), realloc() возвращает NULL?
"""

class Module3_Submodule5_Task9(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.seed)
        self.correct = None
        self.stage = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        self.correct = ["нехватка памяти", "out of memory", "oom", "ошибка выделения", "нет памяти"]
        return TASK_DESCRIPTION

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = ""
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(output, exp)
            )
        ]

    def _compare_default(self, output: str, expected: str) -> bool:
        text = output.lower()
        return any(kw in text for kw in self.correct)

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
            
            if self._compare_default(self.student_solution, ""):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
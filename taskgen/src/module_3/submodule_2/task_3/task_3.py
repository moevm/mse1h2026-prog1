from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem

TASK_DESCRIPTION = """### Тема: Проверка указателя на NULL
**Сложность:** легкая

**Задание:**
Напишите корректное условие для проверки указателя `ptr` на значение `NULL`. Ответ должен содержать оператор `if` и сравнение.
"""

class Module3_Submodule2_Task3(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.seed)
        self.correct = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        self.correct = "if (ptr == NULL)"
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
                compare_func=lambda output, exp: self._compare_default(output, exp)
            )
        ]

    def _compare_default(self, output: str, expected: str) -> bool:
        clean = re.sub(r'\s+', '', output.strip().rstrip(';').lower())
        valid_variants = {
            "if(ptr==null)",
            "if(ptr==0)",
            "if(!ptr)",
            "if(0==ptr)"
        }
        return clean in valid_variants

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
            
            if self._compare_default(self.student_solution, self.correct):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
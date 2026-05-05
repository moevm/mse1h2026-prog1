from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem


class Module3_Submodule2_Task1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.seed)
        self.expected = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        rem = self.seed % 3
        types = ["int", "float", "char"]
        self.type = types[rem]
        self.expected = f"{self.type} *ptr"
        return f"""### Тема: Память и модель памяти

**Сложность:** легкая

**Задача:** Объявите указатель на {self.type}."""

    def compile(self) -> Optional[str]:
        return None  

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=self.expected,
                compare_func=lambda output, exp: self._compare_decl(output, exp)
            )
        ]

    def _compare_decl(self, output: str, expected: str) -> bool:
        clean_out = re.sub(r'\s+', ' ', output.strip().rstrip(';').lower())
        clean_exp = re.sub(r'\s+', ' ', expected.strip().rstrip(';').lower())
        return clean_out == clean_exp

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
            
            if self._compare_decl(self.student_solution, self.expected):
                return True, "OK: Верное объявление."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
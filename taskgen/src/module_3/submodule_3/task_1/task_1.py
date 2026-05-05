from typing import Optional
import random
import re
from src.base_module.base_task import BaseTaskClass, TestItem


class Module3_Submodule3_Task1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.seed)
        self.correct = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        rem = self.seed % 3
        if rem == 0:
            type_str, name_arr = "int", "int_arr"
        elif rem == 1:
            type_str, name_arr = "float", "float_arr"
        else:
            type_str, name_arr = "char", "char_arr"
            
        size = (self.seed % 15) * (self.seed % 100) + 2
        self.correct = f"{type_str} {name_arr}[{size}]"
        
        return f"""###  Тема: Объявление

**Сложность:** легкая

**Задание:** Объявите массив с именем {name_arr} размером {size} элементов типа {type_str}."""

    def compile(self) -> Optional[str]:
        return None  

    def _generate_tests(self):
        expected = self.correct
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
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
            
            if self._compare_decl(self.student_solution, self.correct):
                return True, "OK: Верное объявление."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
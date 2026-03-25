from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem

TASK_DESCRIPTION = "Объявите указатель с именем ptr на переменную типа {type}."

VARIANTS = (
    ("int", "int *ptr"),
    ("float", "float *ptr"),
    ("char", "char *ptr")
)

class Module3_Submodule2_Task1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.rng = random.Random(self.seed)
        self.correct = None
        self.stage = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        if self.seed % 3 == 0:
            self.stage, self.correct = VARIANTS[0]
            return TASK_DESCRIPTION.format(type=self.stage)
        elif self.seed % 3 == 1:
            self.stage, self.correct = VARIANTS[1]
            return TASK_DESCRIPTION.format(type=self.stage)
        else:
            self.stage, self.correct = VARIANTS[2]
            return TASK_DESCRIPTION.format(type=self.stage)

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
                return False, f"FAIL: Ожидалось {self.correct}, получено {self.student_solution}"
        except Exception as e:
            return False, f"FAIL: {str(e)}"
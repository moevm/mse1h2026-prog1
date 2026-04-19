from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask8(BaseTaskClass):
    """Задание №2.8.1"""

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._a, self._b, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        a = rng.randint(1, 1000)
        b = rng.randint(1, 1000)
        
        expected_answer = a + b
        
        return a, b, expected_answer

    
    def generate_task(self):
        self.task_text = (
            f"Дан код:\n"
            f"```c\n"
            f"#define A {self._a}\n"
            f"#define B {self._b}\n"
            f"#define C (A + B)\n"
            f"```\n"
            f"Чему равно выражение C? Напишите результат в виде целого числа."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self) -> tuple[bool, str]:
        student_answer = getattr(self, "student_solution", "").strip()
        expected = str(self._expected_answer)

        try:
            if int(student_answer) == int(expected):
                return True, "OK: Верный ответ."
            return False, f"FAIL: Неверный ответ."
        except ValueError:
            return False, f"FAIL: Ожидалось целое число."
from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask7(BaseTaskClass):
    """Задание №2.7.1"""

    _NAME_PREFIXES = ["SQR", "SQ", "MUL", "X2", "SQUARE"]
    
    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._a, self._b, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        prefix = rng.choice(self._NAME_PREFIXES)
        name = f"{prefix}_{rng.randint(1, 999)}"

        a = rng.randint(1, 1000)
        b = rng.randint(1, 1000)
        
        expected_answer = a + b * a + b
        
        return name, a, b, expected_answer

    
    def generate_task(self):
        self.task_text = (
            f"Дан макрос #define {self._name}(x) x * x. Чему равно {self._name}({self._a} + {self._b})? Напишите результат в виде целого числа."
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
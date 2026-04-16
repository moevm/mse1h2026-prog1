from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask9(BaseTaskClass):
    """Задание №2.9.2"""

    _VAR_NAMES = ["x", "y", "a", "b", "val", "num"]
    _VALUES = ["0", "1", "-1", "NULL", "0xFF"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._var1, self._var2, self._val, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        var1 = rng.choice(self._VAR_NAMES)
        var2 = rng.choice([v for v in self._VAR_NAMES if v != var1])

        val = rng.choice(self._VALUES)
        
        expected_answer = (
            f"#define INIT_{var1.upper()}_{var2.upper()}() \\\n"
            f"    {var1} = {val}; \\\n"
            f"    {var2} = {val}"
        )
        
        return var1, var2, val, expected_answer

    def generate_task(self):
        self.task_text = (
            f"Напишите макрос INIT_{self._var1.upper()}_{self._var2.upper()}(), который инициализирует переменные {self._var1} и {self._var2} значением {self._val}. Используйте многострочный макрос с символом `\\` для продолжения строк."
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
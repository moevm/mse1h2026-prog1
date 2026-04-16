from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask12(BaseTaskClass):
    """Задание №2.12.1"""

    _MACRO_NAMES = ["CONCAT", "PASTE", "JOIN", "MERGE", "CAT", "APPEND"]
    _PREFIXES = ["var", "temp", "data", "num", "arr", "obj"]
    _SUFFIXES = ["_1", "_2", "_ptr", "_end", "123", "_val"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._prefix, self._suffix, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        name = rng.choice(self._MACRO_NAMES)
        prefix = rng.choice(self._PREFIXES)
        suffix = rng.choice(self._SUFFIXES)

        expected_answer = f"{prefix}{suffix}"
        
        return name, prefix, suffix, expected_answer

    def generate_task(self):
        self.task_text = (
            f"Дан макрос #define {self._name}(a, b) a##b. Чему будет равно значение {self._name}({self._prefix}, {self._suffix}) после раскрытия макроса?"
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
        expected = self._expected_answer

        if student_answer == expected:
            return True, "OK: Верный ответ."
        
        if student_answer.replace(" ", "") == expected:
            return False, f"FAIL: Оператор ## склеивает токены без пробелов."
        if student_answer.startswith('"') or student_answer.startswith("'"):
            return False, f"FAIL: Результат раскрытия макроса — это токен (идентификатор/число), а не строка."
        return False, f"FAIL: Неверный ответ."
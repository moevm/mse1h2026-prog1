from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask11(BaseTaskClass):
    """Задание №2.11.1"""

    _MACRO_NAMES = ["STR", "TOSTR", "MKSTR", "STRIFY", "PRINT"]
    _TEXTS = ["hello", "world", "test123", "foo_bar", "abc def", "C_macro"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._text, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        name = rng.choice(self._MACRO_NAMES)
        text = rng.choice(self._TEXTS)

        expected_answer = f'"{text}"'
        return name, text, expected_answer

    def generate_task(self):
        self.task_text = (
            f"Дан макрос #define {self._name}(x) #x. Чему будет равно значение {self._name}({self._text}) после раскрытия макроса? Напишите результат как строковый литерал."
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
        
        if student_answer == expected.replace('"', "'"):
            return False, f"FAIL: В C строковые литералы заключаются в двойные кавычки."
        return False, f"FAIL: Неверный ответ."
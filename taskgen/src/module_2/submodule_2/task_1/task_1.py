from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask1(BaseTaskClass):
    """Задание №2.1.4"""
    _TYPES = (
        "стандартных/системных",
        "собственных",
    )
    
    _ANSWERS = {
        "стандартных/системных": "<>",
        "собственных": "\"\"",
    }

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file_type, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        
        file_type = rng.choice(self._TYPES)
        expected_answer = self._ANSWERS[file_type]
        
        return file_type, expected_answer
    
    def generate_task(self):
        self.task_text = (
            f"Какой синтаксис #include используется для подключения {self._file_type} заголовочных файлов?"
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

        def normalize(s: str) -> str:
            return s.strip().replace('"', '"').replace("'", '"')
        
        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
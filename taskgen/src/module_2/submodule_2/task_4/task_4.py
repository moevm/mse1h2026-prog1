from src.base_module.base_task import BaseTaskClass

class Module2PreprocessorTask4(BaseTaskClass):
    """Задание №2.4.1"""

    _EXPECTED_ANSWER = "нет"
    
    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self.student_solution = ""
    
    def generate_task(self):
        self.task_text = (
            "Будет ли ошибкой `#undef` для несуществующего макроса?"
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self) -> tuple[bool, str]:
        student_answer = getattr(self, "student_solution", "").strip().lower()
        expected = self._EXPECTED_ANSWER.lower()
        
        if student_answer == expected:
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
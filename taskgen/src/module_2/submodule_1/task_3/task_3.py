from src.base_module.base_task import BaseTaskClass
import random

class Module2BuildStagesTask3(BaseTaskClass):
    """Задание №1.3.3"""
    _ACTIONS = (
        "ассемблерная инструкция переводится в машинный код",
        "создается структура объектного файла",
        "формируется таблица символов",
    )

    _EXPECTED_ANSWER = "ассемблирование"

    def _pick_variant(self) -> tuple[str, str]:
        rng = random.Random(self.seed)
        return rng.choice(self._ACTIONS)
    
    def generate_task(self):
        action = self._pick_variant()
        self.task_text = (
            f"На каком этапе сборки {action}?"
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self):
        student_answer = getattr(self, "student_solution", "").strip().lower()
        expected = self._EXPECTED_ANSWER.lower()
        
        if student_answer == expected:
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
from src.base_module.base_task import BaseTaskClass
import random

class Module2BuildStagesTask1(BaseTaskClass):
    _VARIANTS = (
        ("препроцессинга", "-E"),
        ("компиляции", "-S"),
        ("ассемблирования", "-c"),
    )

    def _pick_variant(self) -> tuple[str, str]:
        rng = random.Random(self.seed)
        return rng.choice(self._VARIANTS)
    
    def generate_task(self):
        stage, _expected_flag = self._pick_variant()
        self.task_text = (
            f"Какой флаг gcc останавливает сборку после этапа {stage}?"
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def check(self):
        if getattr(self, "student_solution", "").strip():
            return True, "OK: Заглушка."
        return False, "FAIL: Пустое решение."
from src.base_module.base_task import BaseTaskClass
import random

class Module2_Submodule5_Task2(BaseTaskClass):
    """Задание №5.2.1"""

    _NAMES = ["mylib", "utils", "core", "math", "net", "data", "helper"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        name = rng.choice(self._NAMES)
        expected = f"lib{name}.a"
        return name, expected

    def generate_task(self):
        self.task_text = (
            f"При линковке используется флаг `-l{self._name}`. "
            f"Какое полное имя файла (с префиксом и расширением) будет автоматически искать линковщик в указанных директориях для статической библиотеки?"
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, s: str) -> str:
        return ' '.join(s.lower().split())

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."

        if not norm_student.endswith(".a"):
            return False, "FAIL: Ожидается имя файла с расширением `.a`."
        if self._name not in norm_student:
            return False, f"FAIL: Неверное имя библиотеки."
        return False, f"FAIL: Неверный ответ."
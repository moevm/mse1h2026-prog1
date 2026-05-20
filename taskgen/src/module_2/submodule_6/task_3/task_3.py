from src.base_module.base_task import BaseTaskClass
import random

class Module2_Submodule6_Task3(BaseTaskClass):
    """Задание №6.3.1"""

    _VERSIONS = [11, 17]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._ver, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        ver = rng.choice(self._VERSIONS)
        expected = f"-std=gnu{ver}"
        return ver, expected

    def generate_task(self):
        self.task_text = (
            f"Какой флаг необходимо передать компилятору gcc, чтобы соответствовать стандарту языка C версии {self._ver} и при этом сохранить поддержку расширений GCC? "
            f"Запишите флаг со значением."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, s: str) -> str:
        return ' '.join(s.split())

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."

        if not norm_student.startswith("-std="):
            return False, "FAIL: Ожидается флаг формата `-флаг=значение`."
        if f"gnu{self._ver}" not in norm_student:
            return False, f"FAIL: Для сохранения GNU-расширений используйте префикс `gnu`."
        
        return False, f"FAIL: Неверный ответ."
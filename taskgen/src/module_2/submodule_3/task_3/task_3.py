from src.base_module.base_task import BaseTaskClass
import random


class Module2_Submodule3_Task3(BaseTaskClass):
    """Задание №3.3.1"""

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self.N, self.expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        N = rng.randint(1, 25)
        expected_answer = f"1 {N}"
        return N, expected_answer

    def generate_task(self):
        self.task_text = (
            f"Из каких файлов состоит один логический модуль в языке C согласно стандартной практике организации кода, если в модуле сгенерировано {self.N} файлов реализации? "
            f"Запишите ответ в виде двух чисел через пробел: первое - количество файлов интерфейса, второе - количество файлов реализации."
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
        try:
            parts = student_answer.replace(',', ' ').split()
            if len(parts) != 2:
                return False, "FAIL: Ожидаются ровно два числа через пробел или запятую."
            
            ans1, ans2 = int(parts[0]), int(parts[1])
        except ValueError:
            return False, "FAIL: Ожидаются целые числа."

        if ans1 == 1 and ans2 == self.N:
            return True, "OK: Верный ответ."
        else:
            return False, f"FAIL: Неверный ответ."
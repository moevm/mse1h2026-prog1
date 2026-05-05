from src.base_module.base_task import BaseTaskClass
import random

class Module2ComlpilerFlagsTask4(BaseTaskClass):
    """Задание №6.4.1"""

    _FILES = ["main", "app", "test", "utils", "config", "driver", "entry"]
    _OPTIONS = [
        ("без оптимизации, для отладки", "-O0"),
        ("с локальной оптимизацией", "-O1"),
        ("с оптимизацией баланса скорости и стабильности", "-O2"),
        ("с максимальной оптимизацией производительности", "-O3"),
        ("с оптимизацией по размеру кода", "-Os")
    ]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file, self._opt_desc, self._flag, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        file_name = rng.choice(self._FILES)
        desc, flag = rng.choice(self._OPTIONS)
        expected = f"gcc {flag} {file_name}.c"
        return file_name, desc, flag, expected

    def generate_task(self):
        self.task_text = (
            f"Дан файл {self._file}.c. "
            f"Напишите команду gcc для его компиляции {self._opt_desc}. "
            f"Запишите полную команду."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, cmd: str) -> str:
        return ' '.join(cmd.split())

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."

        if not norm_student.startswith("gcc"):
            return False, "FAIL: Команда должна начинаться с gcc."
        if f"{self._file}.c" not in norm_student:
            return False, f"FAIL: Неверно указано имя файла."
        if self._flag not in norm_student:
            return False, f"FAIL: Неверный флаг оптимизации."
            
        return False, f"FAIL: Неверный ответ."
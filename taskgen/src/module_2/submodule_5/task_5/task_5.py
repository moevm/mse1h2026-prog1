from src.base_module.base_task import BaseTaskClass
import random

class Module2LibrariesTask5(BaseTaskClass):
    """Задание №5.5.1"""

    _SRCS = ["main", "app", "test", "entry", "core", "driver"]
    _NAMES = ["mylib", "utils", "math", "net", "data", "helper"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._src, self._name, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        src = rng.choice(self._SRCS)
        name = rng.choice(self._NAMES)
        expected = f"{src}.c -l{name}"
        return src, name, expected

    def generate_task(self):
        self.task_text = (
            f"Напишите корректную последовательность двух аргументов для команды сборки: исходный файл {self._src}.c и флаг библиотеки `-l{self._name}`. "
            f"Разделяйте аргументы одним пробелом."
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

        if f"{self._src}.c" not in norm_student:
            return False, f"FAIL: Отсутствует исходный файл."
        if f"-l{self._name}" not in norm_student:
            return False, f"FAIL: Отсутствует флаг библиотеки."

        src_pos = norm_student.find(f"{self._src}.c")
        lib_pos = norm_student.find(f"-l{self._name}")
        if src_pos > lib_pos:
            return False, f"FAIL: Неверный порядок."
        return False, f"FAIL: Неверный ответ."
from src.base_module.base_task import BaseTaskClass
import random

class Module2_Submodule5_Task3(BaseTaskClass):
    """Задание №5.3.1"""

    _NAMES = ["mylib", "utils", "core", "math", "net", "data", "helper"]
    _OBJS = ["file_a", "file_b", "module1", "module2", "src1", "src2", "comp"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._obj1, self._obj2, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        obj1, obj2 = rng.sample(self._OBJS, 2)
        name = rng.choice(self._NAMES)
        expected = f"ar rcs lib{name}.a {obj1}.o {obj2}.o"
        return name, obj1, obj2, expected

    def generate_task(self):
        self.task_text = (
            f"Вам необходимо создать статическую библиотеку lib{self._name}.a из объектных файлов {self._obj1}.o и {self._obj2}.o. "
            f"Напишите полную команду с использованием утилиты `ar` (используйте стандартные флаги `rcs`)."
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

        if not norm_student.startswith("ar"):
            return False, "FAIL: Команда должна начинаться с утилиты `ar`."
        if "rcs" not in norm_student:
            return False, "FAIL: Отсутствуют необходимые флаги `rcs`."
        if f"lib{self._name}.a" not in norm_student:
            return False, f"FAIL: Неверное имя библиотеки."
        if f"{self._obj1}.o" not in norm_student or f"{self._obj2}.o" not in norm_student:
            return False, f"FAIL: Отсутствуют или неверно указаны объектные файлы."

        return False, f"FAIL: Неверный ответ."
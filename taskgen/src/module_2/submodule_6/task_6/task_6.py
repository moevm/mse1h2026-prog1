from src.base_module.base_task import BaseTaskClass
import random

class Module2ComlpilerFlagsTask6(BaseTaskClass):
    """Задание №6.6.1"""

    _FILES = ["main", "app", "test", "utils", "config", "driver", "entry"]
    _TYPES = ["address", "undefined"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file, self._type, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        file_name = rng.choice(self._FILES)
        san_type = rng.choice(self._TYPES)
        expected = f"gcc -fsanitize={san_type} -g {file_name}.c"
        return file_name, san_type, expected

    def generate_task(self):
        self.task_text = (
            f"Для поиска ошибок во время выполнения в файле {self._file}.c необходимо включить санитайзер {self._type}. "
            f"Напишите команду gcc, включающую данный санитайзер и отладочную информацию."
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

        if "-fsanitize=" not in norm_student:
            return False, "FAIL: Отсутствует флаг для включения санитайзера."
        if f"-fsanitize={self._type}" not in norm_student:
            return False, f"FAIL: Неверный тип санитайзера."
        if "-g" not in norm_student:
            return False, "FAIL: Отсутствует флаг для отладочной информации."
        if f"{self._file}.c" not in norm_student:
            return False, f"FAIL: Неверно указано имя файла."
        
        return False, f"FAIL: Неверный ответ."
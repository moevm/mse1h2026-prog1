from src.base_module.base_task import BaseTaskClass
import random

class Module2ComlpilerFlagsTask7(BaseTaskClass):
    """Задание №6.7.1"""

    _FILES = ["main", "app", "test", "utils", "config", "driver", "entry"]
    _PROGS = ["output", "my_app", "run", "test_bin", "build"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file, self._prog, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        file_name = rng.choice(self._FILES)
        prog_name = rng.choice(self._PROGS)
        expected = f"gcc -Wall -Wextra -Werror {file_name}.c -o {prog_name}"
        return file_name, prog_name, expected

    def generate_task(self):
        self.task_text = (
            f"Дан файл {self._file}.c. "
            f"Напишите полную команду gcc для сборки исполняемого файла {self._prog}, которая включает базовые и расширенные предупреждения, а также гарантирует прерывание сборки при появлении хотя бы одного предупреждения."
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

        if f"{self._file}.c" not in norm_student:
            return False, f"FAIL: Неверно указано имя исходного файла."
        if "-Wall" not in norm_student:
            return False, "FAIL: Отсутствует флаг базовых предупреждений."
        if "-Wextra" not in norm_student:
            return False, "FAIL: Отсутствует флаг расширенных предупреждений."
        if "-Werror" not in norm_student:
            return False, "FAIL: Отсутствует флаг прерывания при предупреждениях."
        if "-o" not in norm_student or self._prog not in norm_student:
            return False, f"FAIL: Отсутствует или неверно указан флаг `-o` для имени выходного файла."
        
        return False, f"FAIL: Неверный ответ."
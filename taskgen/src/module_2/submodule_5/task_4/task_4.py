from src.base_module.base_task import BaseTaskClass
import random

class Module2LibrariesTask4(BaseTaskClass):
    """Задание №5.4.1"""

    _MAINS = ["main", "app", "program", "test", "entry"]
    _NAMES = ["mylib", "utils", "core", "math", "net", "data", "helper"]
    _PROGS = ["output", "my_app", "run", "test_bin", "build"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._main, self._name, self._prog, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        main = rng.choice(self._MAINS)
        name = rng.choice(self._NAMES)
        prog = rng.choice(self._PROGS)
        expected = f"gcc {main}.c -L. -l{name} -o {prog}"
        return main, name, prog, expected

    def generate_task(self):
        self.task_text = (
            f"Дан файл {self._main}.c и статическая библиотека lib{self._name}.a, находящаяся в текущей директории. "
            f"Напишите команду gcc для сборки исполняемого файла {self._prog}, которая корректно укажет путь поиска библиотек и подключит нужную."
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

        if f"{self._main}.c" not in norm_student:
            return False, f"FAIL: Отсутствует основной файл."
        if "-L." not in norm_student:
            return False, "FAIL: Отсутствует указание пути поиска библиотек (`-L.`)."
        if f"-l{self._name}" not in norm_student:
            return False, f"FAIL: Не подключена библиотека."
        if "-o" not in norm_student:
            return False, "FAIL: Отсутствует флаг `-o` для указания имени выходного файла."
        if self._prog not in norm_student:
            return False, f"FAIL: Неверное имя выходного файла."

        if all(x in norm_student for x in [f"{self._main}.c", "-L.", f"-l{self._name}", "-o", self._prog]):
            return False, f"FAIL: Неверный порядок аргументов."
          
        return False, f"FAIL: Неверный ответ."
from src.base_module.base_task import BaseTaskClass
import random

class Module2LinkingTask6(BaseTaskClass):
    """Задание №4.6.1"""

    _MAINS = ["main", "app", "program", "entry", "test"]
    _PROGS = ["output", "my_app", "run", "test_bin", "build"]
    _FUNCS = ["process_data", "init_system", "calculate", "handle_event", "print_info"]
    _IMPLS = ["impl", "utils", "core", "helpers", "logic"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._main, self._prog, self._func, self._impl, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        main = rng.choice(self._MAINS)
        prog = rng.choice(self._PROGS)
        func = rng.choice(self._FUNCS)
        impl = rng.choice(self._IMPLS)
        expected = f"gcc {main}.c {impl}.c -o {prog}"
        return main, prog, func, impl, expected

    def generate_task(self):
        self.task_text = (
            f"При сборке командой `gcc {self._main}.c -o {self._prog}` возникла ошибка `undefined reference to {self._func}`. Объявление функции находится в заголовке, а ее реализация - в файле {self._impl}.c. "
            f"Напишите корректную команду gcc, которая устранит ошибку линковки."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, cmd: str) -> str:
        """Убирает лишние пробелы, переносы строк и табуляции."""
        return ' '.join(cmd.split())

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."

        if f"{self._main}.c" not in norm_student:
            return False, f"FAIL: Отсутствует основной файл {self._main}.c."
        if f"{self._impl}.c" not in norm_student:
            return False, f"FAIL: Отсутствует файл реализации {self._impl}.c."
        if "-o" not in norm_student:
            return False, "FAIL: Отсутствует флаг `-o` для указания имени исполняемого файла."
        if self._prog not in norm_student:
            return False, f"FAIL: Неверное имя выходного файла. Ожидается {self._prog}."
        
        return False, f"FAIL: Неверный ответ."
from src.base_module.base_task import BaseTaskClass
import random


class Module2BuildStagesTask6(BaseTaskClass):
    """Задание №1.6.2"""
    
    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed) 
        self._name, self._prog = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        
        prefixes = ["main", "calc", "utils", "prog", "test", "app", "program"]
        name = f"{prefixes[rng.randint(0, len(prefixes) - 1)]}{rng.randint(1, 999)}"
        prog = f"prog{rng.randint(1, 999)}"
        
        return name, prog

    def generate_task(self):
        self.task_text = (
            f"Дан файл {self._name}.c. Напишите одну команду gcc для полной сборки в исполняемый файл {self._prog}."
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
        
        expected = f"gcc {self._name}.c -o {self._prog}"

        def normalize(s: str) -> str:
            return " ".join(s.split())
        
        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
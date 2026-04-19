from src.base_module.base_task import BaseTaskClass
import random


class Module2BuildStagesTask5(BaseTaskClass):
    """Задание №1.5.2"""
    
    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed) 
        self.student_solution = ""
        self._name = self._generate_params()

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        prefixes = ["main", "calc", "utils", "prog", "test", "app", "program"]
        return f"{rng.choice(prefixes)}{rng.randint(1, 999)}"

    def generate_task(self):
        self.task_text = (
            f"Дан файл {self._name}.c. Напишите команду gcc, которая создаст объектный файл {self._name}.o без линковки."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self) -> tuple[bool, str]:
        student_answer = self.student_solution.strip()
        expected = f"gcc -c {self._name}.c -o {self._name}.o"
        
        def normalize(s: str) -> str:
            return " ".join(s.split())
        
        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
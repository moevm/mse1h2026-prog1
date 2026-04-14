from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask3(BaseTaskClass):
    """Задание №2.3.2"""

    _NAME_PREFIXES = ["SUM", "ADD", "PLUS", "TOTAL", "COMBINE"]
    
    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        
        prefix = rng.choice(self._NAME_PREFIXES)
        name = f"{prefix}_{rng.randint(1, 999)}"
        expected_answer = f"#define {name}(a, b) ((a) + (b))"
        
        return name, expected_answer
    
    def generate_task(self):
        self.task_text = (
            f"Напишите директиву #define, которая создаёт функциональный макрос {self._name}(a, b), возвращающий сумму аргументов."
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
        expected = self._expected_answer

        def normalize(s: str) -> str:
            return " ".join(s.split())
        
        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
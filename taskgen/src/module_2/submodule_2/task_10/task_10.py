from src.base_module.base_task import BaseTaskClass
import random
import string

class Module2PreprocessorTask10(BaseTaskClass):
    """Задание №2.10.1"""

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._var1, self._var2, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        var1 = rng.choice(string.ascii_lowercase)
        var2 = rng.choice([v for v in string.ascii_lowercase if v != var1])

        expected_answer = (
            f"#define SWAP({var1}, {var2}) \\\n"
            f"    do {{ \\\n"
            f"        int t = ({var1}); \\\n"
            f"        ({var1}) = ({var2}); \\\n"
            f"        ({var2}) = t; \\\n"
            f"    }} while (0)"
        )
        
        return var1, var2, expected_answer

    def generate_task(self):
        self.task_text = (
            f"Напишите безопасный многострочный макрос SWAP({self._var1}, {self._var2}), который обменивает значения переменных через временную переменную и использует конструкцию do {{ ... }} while (0)."
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
        expected = self._expected_answer.strip()

        def normalize(code: str) -> str:
            return ' '.join(code.replace('\\', '').split())

        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
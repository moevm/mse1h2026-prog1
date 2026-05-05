from src.base_module.base_task import BaseTaskClass
import random
import string

class Module2HeaderFileTask4(BaseTaskClass):
    """Задание №3.4.1"""

    _TYPES = ["int", "float", "double", "size_t", "long"]
    _VARS = ["counter", "config_val", "global_id", "max_size", "base_addr", "timeout_ms", "version_code"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._type, self._var, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        var_type = rng.choice(self._TYPES)
        var_name = rng.choice(self._VARS)
        expected = f"extern {var_type} {var_name};"
        return var_type, var_name, expected

    def generate_task(self):
        self.task_text = (
            f"В проекте необходимо объявить глобальную переменную {self._var} типа {self._type} в заголовочном файле так, чтобы она была доступна из других модулей, но память под нее не выделялась."
            f" Напишите корректное объявление этой переменной (заканчивающееся точкой с запятой)."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "").strip()
        
        norm_student = ' '.join(student_ans.split())
        if not norm_student.endswith(';'):
            norm_student += ';'
            
        norm_expected = ' '.join(self._expected_answer.split())

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
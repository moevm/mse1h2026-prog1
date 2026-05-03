from src.base_module.base_task import BaseTaskClass
import random

class Module2LinkingTask5(BaseTaskClass):
    """Задание №4.5.1"""

    _TYPES = ["int", "float", "double"]
    _VARS = ["counter", "global_val", "status_flag", "max_size", "base_addr", "timeout", "limit"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._type, self._var, self._val, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        var_type = rng.choice(self._TYPES)
        var_name = rng.choice(self._VARS)
        val = rng.randint(0, 100)
        expected = f"extern {var_type} {var_name};"
        return var_type, var_name, val, expected

    def generate_task(self):
        self.task_text = (
            f"В проекте возникла ошибка линковки `multiple definition of {self._var}`. Переменная `{self._type} {self._var} = {self._val};` определена в двух разных `*.c` файлах. "
            f"Как правильно изменить строку во втором файле, чтобы оставить ровно одно выделение памяти, а остальные файлы перевести в режим объявления? Запишите исправленную строку."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, code: str) -> str:
        for ch in '{}();=':
            code = code.replace(ch, f' {ch} ')
        return ' '.join(code.split())

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
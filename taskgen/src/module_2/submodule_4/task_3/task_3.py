from src.base_module.base_task import BaseTaskClass
import random

class Module2_Submodule4_Task3(BaseTaskClass):
    """Задание №4.3.1"""

    _TYPES = ["int", "float", "double"]
    _VARS = ["counter", "status", "limit", "offset", "mode", "threshold", "flag"]

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
        expected = f"static {var_type} {var_name} = {val};"
        return var_type, var_name, val, expected

    def generate_task(self):
        self.task_text = (
            f"Напишите определение глобальной переменной {self._var} типа {self._type} со значением {self._val}, которая должна иметь внутреннюю линковку "
            f"(быть видимой только внутри текущего `*.c` файла и не экспортироваться линковщику)."
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
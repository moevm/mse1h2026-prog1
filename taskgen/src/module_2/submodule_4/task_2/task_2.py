from src.base_module.base_task import BaseTaskClass
import random

class Module2_Submodule4_Task2(BaseTaskClass):
    """Задание №4.2.1"""

    _TYPES = ["int", "float", "double"]
    _FILES = ["config", "utils", "globals", "settings", "data"]
    _VARS = ["counter", "max_val", "status_flag", "timeout", "version"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file, self._type, self._var, self._val, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        file_name = rng.choice(self._FILES)
        var_type = rng.choice(self._TYPES)
        var_name = rng.choice(self._VARS)
        val = rng.randint(0, 100)
        expected = f"extern {var_type} {var_name};"
        return file_name, var_type, var_name, val, expected

    def generate_task(self):
        self.task_text = (
            f"В файле {self._file}.c определена глобальная переменная:\n"
            f"{self._type} {self._var} = {self._val};\n\n"
            f"Как правильно объявить ее в заголовочном файле {self._file}.h, "
            f"чтобы другие `*.c` файлы могли обращаться к этой переменной, но при этом не возникло ошибки множественного определения? "
            f"Запишите полное объявление."
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
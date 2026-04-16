from src.base_module.base_task import BaseTaskClass
import random

class Module2HeaderFileTask1(BaseTaskClass):
    """Задание №3.1.1"""

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        templates = ["file_{}", "header{}", "myfile{}"]
        file_name = rng.choice(templates).format(self._seed_local)

        guard = file_name.upper().replace('.', '_').replace(' ', '_') + '_H'
        expected_answer = f"#ifndef {guard}\n#define {guard}\n#endif"
        
        return file_name, expected_answer

    def generate_task(self):
        self.task_text = (
            f"Для заголовочного файла {self._file}.h напишите классическую защиту от повторного включения (include guard) в 3 строки. Используйте любое валидное имя макроса для include guard."
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
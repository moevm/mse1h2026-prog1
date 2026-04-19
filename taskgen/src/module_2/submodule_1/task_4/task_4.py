from src.base_module.base_task import BaseTaskClass
import random


class Module2BuildStagesTask4(BaseTaskClass):
    """Задание №1.4.2"""
    
    def _generate_params(self):
        rng = random.Random(self.seed)
        n = rng.randint(2, 4)
        files = [f"file{i+1}" for i in range(n)]
        prog = f"prog{rng.randint(1, 999)}"
        
        return n, files, prog

    def generate_task(self):
        n, files, prog = self._generate_params()
        
        files_str = ", ".join(f"{f}.c" for f in files)
        self.task_text = (
            f"Даны {n} уже скомпилированных файла {files_str}. "
            f"Напишите команду линковки в {prog}."
        )
        
        obj_files = " ".join(f"{f}.o" for f in files)
        self._expected_answer = f"gcc {obj_files} -o {prog}"

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self) -> tuple[bool, str]:
        student_answer = getattr(self, "student_solution", "").strip()
        expected = getattr(self, "_expected_answer", "")
        
        n, files, prog = self._generate_params()
        obj_files = " ".join(f"{f}.o" for f in files)
        expected = f"gcc {obj_files} -o {prog}"
        
        def normalize(s: str) -> str:
            return " ".join(s.split())
        
        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
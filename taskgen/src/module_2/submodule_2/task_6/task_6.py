from src.base_module.base_task import BaseTaskClass
import random

class Module2PreprocessorTask6(BaseTaskClass):
    """Задание №2.6.1"""

    _NAME_PREFIXES = ["VAL", "NUM", "X", "Y", "Z", "A", "B", "V"]

    _RESULT_PREFIXES = ["RES", "OUT", "R", "RESULT", "ANS", "RET"]
    
    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._name, self._result, self._t1, self._t2, self._a, self._b, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        name_prefix = rng.choice(self._NAME_PREFIXES)
        name = f"{name_prefix}_{rng.randint(1, 999)}"
        result_prefix = rng.choice(self._RESULT_PREFIXES)
        result = f"{result_prefix}_{rng.randint(1, 999)}"
        
        t2 = rng.randint(10, 500)
        t1 = rng.randint(t2 + 10, t2 + 500)
        a = rng.randint(1, 1000)
        b = rng.randint(1, 1000)
        
        expected_answer = (
            f"#if {name} > {t1}\n"
            f"#define {result} {a}\n"
            f"#elif {name} > {t2}\n"
            f"#define {result} {b}\n"
            f"#else\n"
            f'#error "Invalid value"\n'
            f"#endif"
        )
        
        return name, result, t1, t2, a, b, expected_answer
    
    def generate_task(self):
        self.task_text = (
            f"Напишите директивы: если {self._name} строго больше {self._t1} - определить макрос {self._result} как {self._a}, если больше {self._t2} - как {self._b}, иначе - выдать ошибку препроцессора с текстом \"Invalid value\"."
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
            lines = s.splitlines()
            normalized_lines = [" ".join(line.split()) for line in lines]
            return "\n".join(normalized_lines)
        
        if normalize(student_answer) == normalize(expected):
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
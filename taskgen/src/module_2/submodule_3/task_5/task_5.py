from src.base_module.base_task import BaseTaskClass
import random

class Module2HeaderFileTask5(BaseTaskClass):
    """Задание №3.5.1"""

    _TYPES = ["int", "float", "double"]
    _FUNCS = ["square", "calc_sq", "pow2", "sqr", "get_sq"]
    _ARGS = ["x", "val", "n", "num", "v"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._type, self._func, self._arg, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        t = rng.choice(self._TYPES)
        f = rng.choice(self._FUNCS)
        a = rng.choice(self._ARGS)
        expected = f"static inline {t} {f}({t} {a}) {{ return {a} * {a}; }}"
        return t, f, a, expected

    def generate_task(self):
        self.task_text = (
            f"Напишите безопасное для заголовочного файла определение функции {self._func}, которая принимает аргумент {self._arg} типа {self._type} и возвращает его квадрат."
            f"Используйте модификаторы, которые предотвратят ошибку множественного определения (multiple definition) при подключении заголовка в несколько `*.c` файлов."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, code: str) -> str:
        for ch in '{}();':
            code = code.replace(ch, f' {ch} ')
        normalized = ' '.join(code.split())
        if not normalized.endswith(';'):
            normalized = normalized.rstrip() + ';'
        return normalized

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."

        if "static" not in norm_student or "inline" not in norm_student:
            return False, "FAIL: Для безопасного определения в .h необходимы модификаторы `static inline`."
        
        target_logic = f"return {self._arg} * {self._arg}"
        if target_logic.replace(' ', '') not in norm_student.replace(' ', ''):
            return False, f"FAIL: Неверная логика вычисления."
        
        if self._type not in norm_student:
            return False, f"FAIL: Неверный или пропущенный тип данных {self._type}."

        return False, f"FAIL: Синтаксическая ошибка или неверный формат ответа."
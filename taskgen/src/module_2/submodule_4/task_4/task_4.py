from src.base_module.base_task import BaseTaskClass
import random

class Module2LinkingTask4(BaseTaskClass):
    """Задание №4.4.1"""

    _FILES = ["utils", "helpers", "math_ops", "core", "driver"]
    _FUNCS = ["calc_mod", "process_val", "get_rem", "compute", "handle_num"]
    _ARGS = ["x", "val", "n", "input", "data"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._file, self._func, self._arg, self._num, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        file_name = rng.choice(self._FILES)
        func_name = rng.choice(self._FUNCS)
        arg_name = rng.choice(self._ARGS)
        num_val = rng.randint(2, 100)
        expected = f"static int {func_name}(int {arg_name}) {{ return (2 * {arg_name}) % {num_val}; }}"
        return file_name, func_name, arg_name, num_val, expected

    def generate_task(self):
        self.task_text = (
            f"В модуле с исполнительным файлом {self._file}.c требуется создать вспомогательную функцию {self._func}, принимающую параметр {self._arg} целочисленного типа и возвращающую остаток от удвоенного значения {self._arg} при делении на {self._num}. "
            f"Функция должна быть видна только внутри текущего `*.c` файла и не экспортироваться линковщику. Напишите ее полное определение."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def _normalize(self, code: str) -> str:
        for ch in '{}();=%+*-/':
            code = code.replace(ch, f' {ch} ')
        return ' '.join(code.split())

    def check(self) -> tuple[bool, str]:
        student_ans = getattr(self, "student_solution", "")
        norm_student = self._normalize(student_ans)
        norm_expected = self._normalize(self._expected_answer)

        if norm_student == norm_expected:
            return True, "OK: Верный ответ."

        if "static" not in norm_student:
            return False, "FAIL: Функция должна иметь внутреннюю линковку (используйте модификатор static)."
        if "int" not in norm_student:
            return False, "FAIL: Параметр и возвращаемое значение должны быть целочисленного типа."
        
        logic_check = f"2 * {self._arg}"
        if logic_check.replace(" ", "") not in norm_student.replace(" ", ""):
            return False, f"FAIL: Неверная формула вычисления. Ожидается удвоение аргумента {self._arg}."
        if f"% {self._num}" not in norm_student.replace(" ", ""):
            return False, f"FAIL: Неверный делитель. Ожидается остаток от деления на {self._num}."
        
        return False, f"FAIL: Неверный ответ."
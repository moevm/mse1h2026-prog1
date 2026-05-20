from src.base_module.base_task import BaseTaskClass
import random

class Module2_Submodule4_Task1(BaseTaskClass):
    """Задание №4.1.1"""

    _VARS = ["counter", "global_val", "status_flag", "max_size", "base_addr"]
    _FUNCS = ["init", "process", "handle_event", "calc", "log_msg"]
    _MACROS = ["CONFIG", "MAX_BUF", "DEBUG", "VERSION", "FLAG"]

    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._var, self._func, self._val, self._MAC, self._num, self._expected_answer = self._generate_params()
        self.student_solution = ""

    def _generate_params(self):
        rng = random.Random(self._seed_local)
        var = rng.choice(self._VARS)
        func = rng.choice(self._FUNCS)
        val = rng.randint(0, 100)
        mac = rng.choice(self._MACROS)
        num = rng.randint(0, 100)
        expected = f"{var}, {func}"
        return var, func, val, mac, num, expected

    def generate_task(self):
        self.task_text = (
            f"Дан фрагмент кода:\n"
            f"```c\n"
            f"  int {self._var} = {self._val};\n"
            f"  void {self._func}(int a);\n"
            f"  #define {self._MAC} {self._num}\n"
            f"```\n"
            f"Перечислите через запятую все имена, которые линковщик будет обрабатывать как символы (в порядке появления в коде)."
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
        parts = [p.strip() for p in student_ans.replace(',', ' ').split() if p]
        expected_parts = [self._var, self._func]

        if parts == expected_parts:
            return True, "OK: Верный ответ."
        return False, f"FAIL: Неверный ответ."
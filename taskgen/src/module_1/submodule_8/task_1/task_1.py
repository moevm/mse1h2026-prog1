from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const(seed):
    return seed % 20 + 10


def generate_flag(seed):
    if seed % 2 == 0: return "o"
    return "x"


def check_answer(const, flag):
    if flag == "o": return oct(const)[2:]
    else: return hex(const)[2:]


class Module_1_Submodule_8_task_1(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const = generate_const(self.seed)
        self.flag = generate_flag(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return(
            f"Что выведет следующий фрагмент программы:\n"
            f"int a = {self.const};\n"
            f"printf(\"%{self.flag}\", a);\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.const, self.flag)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(output.strip(), exp)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.student_solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self):
        try:
            self.generate_task()
            expected = check_answer(self.const, self.flag)
            student = self.student_solution.strip()
            if self._compare_default(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_src_name(seed: int) -> str:
    return f"hello_{seed % 1000}"


def check_answer(src_name: str) -> str:
    return f"gcc {src_name}.c -o {src_name} || gcc -o {src_name} {src_name}.c"


def compare_answer(output: str, expected: str) -> bool:
    normalized_output = " ".join(output.strip().split())
    variants = [" ".join(item.strip().split()) for item in expected.split("||")]
    return normalized_output in variants


class Module_1_Submodule_1_task_1(BaseTaskClass):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.src_name = generate_src_name(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (
            f"В текущей директории находится файл {self.src_name}.c, "
            f"который выводит на экран сообщение \"Hello, world!\". "
            f"Какую команду нужно ввести для компиляции, "
            f"чтобы исполняемый файл назывался {self.src_name}."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.src_name)

        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: compare_answer(output, exp)
            )
        ]

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def run_solution(self, test: TestItem):
        student_answer = self.student_solution
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def check(self):
        try:
            self.generate_task()
            expected = check_answer(self.src_name)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, f"FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"

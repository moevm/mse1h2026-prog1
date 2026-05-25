from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_s1(seed: int) -> str:
    return "//" if seed % 2 == 1 else ""


def generate_s2(seed: int) -> str:
    return "//" if (seed // 2) % 2 == 1 else ""


def generate_s3(seed: int) -> str:
    return "//" if (seed // 4) % 2 == 1 else ""


def generate_s4(seed: int) -> str:
    return "//" if (seed // 8) % 2 == 1 else ""


def check_answer(s1: str, s2: str, s3: str, s4: str) -> str:
    result = []
    if s1 == "":
        result.append("a")
    if s2 == "":
        result.append("b")
    if s3 == "":
        result.append("c")
    if s4 == "":
        result.append("d")
    return "".join(result)


def compare_answer(output: str, expected: str) -> bool:
    normalized_output = "".join(output.strip().split())
    normalized_expected = "".join(expected.strip().split())
    return normalized_output == normalized_expected


class Module_1_Submodule_1_task_8(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S1 = generate_s1(self.seed)
        self.S2 = generate_s2(self.seed)
        self.S3 = generate_s3(self.seed)
        self.S4 = generate_s4(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        code = (
            "#include <stdio.h>\n\n"
            "int main()\n"
            "{\n"
            f"{self.S1}    printf(\"a\");\n"
            f"{self.S2}    printf(\"b\");\n"
            f"{self.S3}    printf(\"c\");\n"
            f"{self.S4}    printf(\"d\");\n"
            "    return 0;\n"
            "}"
        )
        return (
            "В программу вставлены комментарии. Определите, какие символы появятся на экране, "
            "и запишите их без пробелов и других разделителей (например ab).\n\n"
            + code
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.S1, self.S2, self.S3, self.S4)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: compare_answer(output, exp)
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
            expected = check_answer(self.S1, self.S2, self.S3, self.S4)
            student = self.student_solution.strip()
            if compare_answer(student, expected):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
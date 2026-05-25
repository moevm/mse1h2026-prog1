from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem

VALID_SIGNATURES = [
    "int main(void)",
    "int main()",
    "int main(int argc, char *argv[])",
]

INVALID_SIGNATURES = [
    "int main(int argc)",
    "int main(char *argv[])",
    "int main(int argv, char *argc[])",
    "int main(int argc, char *argv)",
    "int main(int)",
    "int main(char)",
    "int main(int argv, char *argc)",
    "int main(int argc, char argv[])",
    "int main(int argc, char argv)",
    "int main(int argv, char argc[])",
    "int main(int argv, char argc)",
    "void main(void)",
    "long main(void)",
    "char main(int argc, char *argv[])",
    "void main(int argc, char argv[])",
]

def build_signatures(seed: int) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    valid_count = rng.randint(1, 3)
    chosen_valid = rng.sample(VALID_SIGNATURES, valid_count)
    chosen_invalid = rng.sample(INVALID_SIGNATURES, 8 - valid_count)
    combined = [(sig, True) for sig in chosen_valid] + [(sig, False) for sig in chosen_invalid]
    rng.shuffle(combined)
    signatures = [sig for sig, _ in combined]
    right_indices = [i + 1 for i, (_, is_valid) in enumerate(combined) if is_valid]
    return signatures, right_indices


def check_answer(right_indices: list[int]) -> str:
    return " ".join(map(str, right_indices))


class Module_1_Submodule_1_task_2(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signatures, self.right_indices = build_signatures(self.seed)
        self.student_solution = ""

    def generate_task(self) -> str:
        lines = [
            "Какие из представленных ниже сигнатур функции main являются корректными?",
            "Укажите номера правильных вариантов через пробел без других знаков.",
        ]
        lines.extend(f"{idx}. {signature}" for idx, signature in enumerate(self.signatures, start=1))
        return "\n".join(lines)

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.right_indices)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(
                    " ".join(output.strip().split()),
                    exp
                )
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
            expected = check_answer(self.right_indices)
            student = self.student_solution.strip()
            if " ".join(student.split()) == expected:
                return True, "OK: Верный ответ."
            else:
                return False, f"FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
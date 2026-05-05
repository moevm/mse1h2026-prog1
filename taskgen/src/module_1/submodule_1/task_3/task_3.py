from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


FLAGS = ["-p", "-n", "-v", "--help", "--mode", "--seed", "-o"]


def generate_argv(seed: int) -> list[str]:
    rng = random.Random(seed)
    argv = ["./a.out"]
    extra_count = rng.randint(2, 6)

    for i in range(extra_count):
        if i % 2 == 0 and rng.random() < 0.5:
            argv.append(rng.choice(FLAGS))
        else:
            argv.append(str(rng.randint(0, 99)))
    return argv


def check_answer(argv: list[str]) -> str:
    return str(len(argv))


class Module_1_Submodule_1_task_3(BaseTaskClass):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.argv = generate_argv(self.seed)

    def generate_task(self) -> str:
        argv_text = " ".join(self.argv)
        return (
            "Чему равен argc, если argv = "
            f"{argv_text}?\n"
            "int main(int argc, char *argv[]);"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.argv)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(
                    output.strip(),
                    exp
                )
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

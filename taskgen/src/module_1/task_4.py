from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


class PrintProgramTask(BaseTaskClass):
    """
    Задание: Что выведет программа при запуске?
    #include <stdio.h>

    int main()
    {
        printf("<text>\n");
        return 0;
    }
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        rng = random.Random(self.seed)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        src_name = ''.join(rng.choices(letters, k=rng.randint(5, 8)))
        self.text = src_name

    def generate_task(self) -> str:
        return(
            "Что выведет программа при запуске?\n"
            "#include <stdio.h>\n\n"
            "int main()\n"
            "{\n"
            f'    printf("{self.text}\\n");\n'
            "    return 0;\n"
            "}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = self.text

        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, expected: self._compare_default(
                    output.strip(), expected)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def _compare_default(self, output: str, expected: str) -> bool:
        return output.strip() == expected.strip() or output.strip() == (expected.strip() + "\n")
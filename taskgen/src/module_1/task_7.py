from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


class ErrorPlaceTask(BaseTaskClass):
    """
    Задание: В какой строке программы допущена ошибка?
    #include <stdio.h>

    int main()
    {
        printf("Hello world!\n");
        return 0;
    }
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        error_line = [1, 3, 5, 6]
        self.line_of_error = error_line[self.seed % 4]

    def generate_task(self) -> str:
        programs = []
        programs.append(" #include <stdio>\n\n int main()\n {\n     printf(\"Hello world!\\n\");\n     return 0;\n }")
        programs.append(" #include <stdio.h>\n\n int main(\n {\n     printf(\"Hello world!\\n\");\n     return 0;\n }")
        programs.append(" #include <stdio.h>\n\n int main()\n {\n     printf(\"Hello world!\\n\")\n     return 0;\n }")
        programs.append(" #include <stdio.h>\n\n int main()\n {\n     printf(\"Hello world!\\n\");\n     return 1;\n }")
        return (
            "Задание: В какой строке программы допущена ошибка?\n"
            f"{programs[self.seed % 4]}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = str(self.line_of_error)

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
        return output.strip() == expected.strip()
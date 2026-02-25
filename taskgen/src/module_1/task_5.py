from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


class TypeOfFunctionTask(BaseTaskClass):
    """
    Задание: Какой тип возвращаемого значения у функции?
    <тип> function(<тип> element)
    {
        return element;
    }
    или
    <тип> function();
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        types = ["int", "double", "float", "char"]
        type_of_function = types[self.seed % 4]
        self.type_of_function = type_of_function

    def generate_task(self) -> str:
        if self.seed % 2 == 0:
            return(
                "Какой тип возвращаемого значения у функции?\n"
                f"{self.type_of_function} function({self.type_of_function} element)\n"
                "{\n"
                "    return element;\n"
                "}"
            )
        else:
            return(
                 "Какой тип возвращаемого значения у функции?\n"
                 f"{self.type_of_function} function();"
            )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = self.type_of_function

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
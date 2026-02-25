from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


class DefFunctionTask(BaseTaskClass):
    """
    Задание: Напишите объявление функции по ее определению.
    <type> function(<type1> obj1, <type2> obj2, <type3> obj3)
    {
        <type> obj4 = <answer>;
        return <answer>;
    }
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.types = ["int", "double", "float", "char"]
        self.obj = [7, 3.141592653589793, 3.14, "a"]
        user_types = []
        for i in range(4):
            index = (self.seed + i) * 3 % 4
            user_types.append(index)
        self.user_types = user_types

    def generate_task(self) -> str:
        return (
            "Задание: Напишите объявление функции по ее определению.\n"
            f"{self.types[self.user_types[0]]} function({self.types[self.user_types[1]]} obj1, "
            f"{self.types[self.user_types[2]]} obj2, {self.types[self.user_types[3]]} obj3)\n"
            "{\n"
            f"    {self.types[self.user_types[0]]} obj = {self.obj[self.user_types[0]]};\n"
            "    return obj;\n"
            "}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        part_1 = f"{self.types[self.user_types[0]]} function({self.types[self.user_types[1]]} obj1, "
        part_2 = f"{self.types[self.user_types[2]]} obj2, {self.types[self.user_types[3]]} obj3);"
        expected = part_1 + part_2

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
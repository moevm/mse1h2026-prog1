from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


TYPES = ["int", "char", "long", "short", "float", "double"]


def generate_params(seed: int) -> tuple[str, str, str, str, str]:
    rng = random.Random(seed)
    ret_type = rng.choice(TYPES)
    type1 = rng.choice(TYPES)
    type2 = rng.choice(TYPES)
    type3 = rng.choice(TYPES)

    if ret_type in ("float", "double"):
        answer = f"{rng.randint(1, 99)}.{rng.randint(0, 9)}"
    elif ret_type == "char":
        answer = f"'{chr(ord('a') + seed % 26)}'"
    else:
        answer = str(rng.randint(1, 99))
    return ret_type, type1, type2, type3, answer


def check_answer(ret_type: str, type1: str, type2: str, type3: str) -> str:
    return f"{ret_type} function({type1} obj1, {type2} obj2, {type3} obj3);"


def normalize_decl(s: str) -> str:
    return " ".join(s.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace(";", " ; ").split())


class Module_1_Submodule_1_task_4(BaseTaskClass):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ret_type, self.type1, self.type2, self.type3, self.answer = generate_params(self.seed)

    def generate_task(self) -> str:
        return (
            "Напишите объявление функции по ее определению.\n"
            f"{self.ret_type} function({self.type1} obj1, {self.type2} obj2, {self.type3} obj3)\n"
            "{\n"
            f"    {self.ret_type} obj4 = {self.answer};\n"
            "    return obj4;\n"
            "}"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.ret_type, self.type1, self.type2, self.type3)
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: normalize_decl(output) == normalize_decl(exp)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const1(seed):
    return seed 


def generate_const2(seed): 
    return seed // 7 + seed % 7
    
    
def generate_operation1(seed):
    if seed % 2 == 0: return "++"
    return "--"


def generate_operation2(seed):
    if seed % 4 == 0: return "+"
    if seed % 4 == 1: return "-"
    if seed % 4 == 2: return "*"
    return "/"
    
    
#def generate_operation3(seed):
#    if (seed // 5) % 2 == 0: return "++"
#    return "--"
    
    
# def generate_operation4(seed):
#     if (seed // 3) % 4 == 0: return "+"
#     if (seed // 3) % 4 == 1: return "-"
#     if (seed // 3) % 4 == 2: return "*"
#     return "/"
    
    
# def generate_operation5(seed):
#     if (seed // 5) % 4 == 0: return "+"
#     if (seed // 5) % 4 == 1: return "-"
#     if (seed // 5) % 4 == 2: return "*"
#     return "/"
 

def check_answer(const1, const2, operation1, operation2, operation3, operation4, operation5):
    if operation1 == "++": const1 += 1
    else: const1 -= 1
    if operation2 == "+": answer = const1 + const2
    elif operation2 == "-": answer = const1 - const2
    elif operation2 == "*": answer = const1 * const2
    else: answer = const1 // const2
    if operation3 == "++": const2 += 1
    else: const2 -= 1
    if operation5 == "+": part = const1 + const2
    elif operation5 == "-": part = const2 - const1
    elif operation5 == "*": part = const1 * const2
    else: part = const2 // const1
    if operation4 == "+": answer += part
    elif operation4 == "-": answer -= part
    elif operation4 == "*": answer *= part
    else: answer //= part
    return str(answer)


class Module_1_Submodule_5_task_1(BaseTaskClass):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.const1 = generate_const1(self.seed)
        self.const2 = generate_const2(self.seed)
        self.operation1 = generate_operation1(self.seed)
        self.operation2 = generate_operation2(self.seed)
        self.operation3 = generate_operation1(self.seed // 5)
        self.operation4 = generate_operation2(self.seed // 3)
        self.operation5 = generate_operation2(self.seed // 5)

    def generate_task(self) -> str:
        return(
            f"Чему равно значение переменной c после вычисления данного фрагмента кода?\n"
            f"int a = {self.const1};\n"
            f"int b = {self.const2};\n"
            f"int c = {self.operation1}a {self.operation2} b{self.operation3};\n"
            f"c {self.operation4}= b {self.operation5} a;\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.const1, self.const2, self.operation1, self.operation2, self.operation3, self.operation4, self.operation5)
        
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda output, exp: self._compare_default(
                    output.strip(), self.file_name)
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected


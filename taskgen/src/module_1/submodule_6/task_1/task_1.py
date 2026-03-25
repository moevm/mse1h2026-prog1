from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_const2(seed): 
    return seed % 4 + 3
    

def generate_const3(seed):
    if seed % 4 == 0 or seed % 4 == 1: return 1
    return -1


def generate_const1(seed):
    return (seed % 10 + 1) * generate_const2(seed) + generate_const3(seed)
    
    
def generate_operation(seed):
    if seed % 4 == 0: return "++a"
    if seed % 4 == 1: return "a++"
    if seed % 4 == 2: return "--a"
    return "a--"
 

def check_answer(const1, const2, operation):
    a = const1
    answer = []
    while a % const2:
        if seed % 4 == 0: 
            a += 1
            answer.append(a)
        if seed % 4 == 1: 
            answer.append(a)
            a += 1
        if seed % 4 == 2:
            a -= 1
            answer.append(a)
        else:
            answer.append(a)
            a -= 1
    return " ".join(answer)
            

class Module_1_Submodule_6_task_1(BaseTaskClass):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.const1 = generate_const1(self.seed)
        self.const2 = generate_const2(self.seed)
        self.const3 = generate_const3(self.seed)
        self.operation = generate_operation(self.seed)

    def generate_task(self) -> str:
        return(
            f"Что выведет программа:\n"
            f"#include <stdio.h>\n\n"
            "int main()\n{\n"
            f"    int a = {self.const1};\n"
            f"    while (a % {self.const2})\n"
            "    {\n"
            f"        printf(\"%d \", {self.operation});\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = check_answer(self.const1, self.const2, self.operation)
        
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


from typing import Optional
import re
from src.base_module.base_task import BaseTaskClass, TestItem


class Module3_Submodule6_Task1(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected = None
        self.student_solution = "" 

    def generate_task(self) -> str:
        rem = self.seed % 4
        configs = [
            ("fp", "int", "int, int"),
            ("calc", "float", "double, char"),
            ("handler", "void", "const char*, int"),
            ("transform", "char*", "int, float")
        ]
        ptr_name, ret_type, param_types = configs[rem]
        self.expected = f"{ret_type} (*{ptr_name})({param_types});"
        
        return f"""### Тема: Синтаксис указателя на функцию
**Сложность:** легкая

**Задание:**
Объявите указатель на функцию с именем `{ptr_name}`, которая принимает параметры типов `{param_types}` и возвращает значение типа `{ret_type}`.
В ответе напишите только строку объявления, завершающуюся `;`. Имена параметров указывать не обязательно.
"""

    def compile(self) -> Optional[str]:
        return None  

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=self.expected,
                compare_func=lambda output, exp: self._compare_decl(output, exp)
            )
        ]

    def _compare_decl(self, output: str, expected: str) -> bool:
        clean_out = re.sub(r'\s+', ' ', output.strip().rstrip(';').lower())
        clean_exp = re.sub(r'\s+', ' ', expected.strip().rstrip(';').lower())

        if clean_out == clean_exp:
            return True

        rem = self.seed % 4
        configs = [
            ("fp", "int", "int, int"),
            ("calc", "float", "double, char"),
            ("handler", "void", "const char*, int"),
            ("transform", "char*", "int, float")
        ]
        ptr_name, ret_type, param_types = configs[rem]
        
        ret_esc = re.escape(ret_type)
        name_esc = re.escape(ptr_name)
        type_parts = [t.strip() for t in param_types.split(',')]
        type_patterns = [rf"{re.escape(t)}(?:\s+\w+)?" for t in type_parts]
        params_regex = r"\s*,\s*".join(type_patterns)
        
        full_pattern = rf"^{ret_esc}\s*\*\s*\(\s*{name_esc}\s*\)\s*\(\s*{params_regex}\s*\)$"
        return bool(re.match(full_pattern, clean_out))

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
            
            if self._compare_decl(self.student_solution, self.expected):
                return True, "OK: Верное объявление."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
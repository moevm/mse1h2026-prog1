from typing import Optional
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from base_module.base_task import *

from constants import *

# Простые задачи (2.1.х)


class Task2_1_1_IncludeAngle(BaseTaskClass):
    """2.1.1: Где препроцессор ищет файл при #include <name.h>?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"header{seed}"
        self.correct_answer: str = "в системных директориях"

    def generate_task(self) -> str:
        return f"Где препроцессор ищет файл при #include <{self.file_name}.h>?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_text
        )]

    def _compare_text(self, output: str, expected: str) -> bool:
        return output.strip().lower() == expected.strip().lower()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )
        

class Task2_1_2_IncludeQuotes(BaseTaskClass):
    """2.1.2: Где препроцессор сначала ищет файл при #include "name.h"?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"header{seed}"
        self.correct_answer: str = "в текущей директории"

    def generate_task(self) -> str:
        return f"Где препроцессор сначала ищет файл при #include \"{self.file_name}.h\"?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_text
        )]

    def _compare_text(self, output: str, expected: str) -> bool:
        return output.strip().lower() == expected.strip().lower()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


class Task2_1_3_IncludeSyntax(BaseTaskClass):
    """2.1.3: Какой синтаксис #include для стандартных/собственных заголовков?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.header_type: Optional[str] = None
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False

    def generate_task(self) -> str:
        if self._task_generated: 
            return self._cached_task_text

        self.header_type = self._rng.choice(["стандартных/системных", "собственных"])
        if self.header_type == "стандартных/системных":
            self.correct_answer = "<>"
        else:
            self.correct_answer = "\"\""

        self._cached_task_text = f"Какой синтаксис #include используется для подключения {self.header_type} заголовочных файлов?"  
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_syntax
        )]

    def _compare_syntax(self, output: str, expected: str) -> bool:
        return output.strip() == expected.strip()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


class Task2_1_4_MacroType(BaseTaskClass):
    """2.1.4: Является ли #define объектоподобным или функциональным макросом?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.macro_name: str = f"MACRO{seed}"
        self.macro_value: Optional[str] = None
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False

    def generate_task(self) -> str:
        if self._task_generated: 
            return self._cached_task_text

        is_functional = self._rng.choice([True, False])
        if is_functional:
            self.macro_value = f"(x) ((x)*(x))"
            self.correct_answer = "функциональный"
        else:
            self.macro_value = str(self._rng.randint(1, 100))
            self.correct_answer = "объектоподобный"
        
        self._cached_task_text = f"Является ли #define {self.macro_name} {self.macro_value} объектоподобным или функциональным макросом?"  
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_text
        )]

    def _compare_text(self, output: str, expected: str) -> bool:
        return output.strip().lower() == expected.strip().lower()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


class Task2_1_5_DirectiveSemicolon(BaseTaskClass):
    """2.1.5: Заканчиваются ли директивы препроцессора точкой с запятой? / Будет ли ошибкой #undef для несуществующего макроса?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.question_type: int = seed % 2
        self.correct_answer: str = "нет"

    def generate_task(self) -> str:
        if self.question_type == 0:
            return "Заканчиваются ли директивы препроцессора точкой с запятой?"
        else:
            return "Будет ли ошибкой #undef для несуществующего макроса?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_yes_no
        )]

    def _compare_yes_no(self, output: str, expected: str) -> bool:
        return output.strip().lower() in ["нет", "no", "нельзя", "false"]

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


class Task2_1_6_DirectiveMeaning(BaseTaskClass):
    """2.1.6: Что делает #ifdef/#ifndef/#if/#undef?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.directive: Optional[str] = None
        self.macro_name: str = f"MACRO{seed}"
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        
    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.directive = self._rng.choice(list(DIRECTIVE_DESCRIPTIONS.keys()))
        self.correct_answer = DIRECTIVE_DESCRIPTIONS[self.directive]
        
        self._cached_task_text = f"Что делает {self.directive} {self.macro_name}?\n\na) включает блок кода, если макрос {self.macro_name} определён\nb) отменяет ранее определённый макрос {self.macro_name}\nc) включает блок кода, если макрос {self.macro_name} не определён\nd) включает блок кода, если выражение {self.macro_name} не равно нулю"  
        self._task_generated = True 
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_directive_meaning
        )]

    def _compare_directive_meaning(self, output: str, expected: str) -> bool:
        out_lower = output.lower()
        exp_lower = expected.lower()
        
        if "определён" in exp_lower:
            return "определён" in out_lower
        elif "не определён" in exp_lower:
            return "не определён" in out_lower
        elif "отменяет" in exp_lower:
            return "отменяет" in out_lower
        elif "не равно нулю" in exp_lower:
            return "не равно нулю" in out_lower or "!= 0" in out_lower
        return out_lower == exp_lower

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


# Средние задачи (2.2.х)


class Task2_2_1_MacroExpansion(BaseTaskClass):
    """2.2.1: Чему равно выражение C после раскрытия макросов?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.a_val: int = self._rng.randint(1, 50)
        self.b_val: int = self._rng.randint(1, 50)
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.correct_answer = str(self.a_val + self.b_val)
        
        code = f"""#define A {self.a_val}
#define B {self.b_val}
#define C (A + B)"""
        
        return f"Дан код:\n```c\n{code}\n```\nЧему равно выражение C?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_number
        )]

    def _compare_number(self, output: str, expected: str) -> bool:
        return output.strip() == expected.strip()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


class Task2_2_2_DefineDirective(BaseTaskClass):
    """2.2.2: Напишите директиву #define для макроса"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.macro_name: str = f"MACRO{seed}"
        self.macro_type: int = 0
        self.macro_value: int = 0
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False 

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.macro_type = self._rng.randint(0, 3)
        self.macro_value = self._rng.randint(1, 100)
        
        if self.macro_type == 0:
            self.correct_answer = f"#define {self.macro_name} {self.macro_value}"
            desc = f"объектоподобный макрос {self.macro_name} со значением {self.macro_value}"
        elif self.macro_type == 1:
            self.correct_answer = f"#define {self.macro_name}(x) ((x) * 2)"
            desc = f"функциональный макрос {self.macro_name}(x), возвращающий удвоенное значение аргумента"
        elif self.macro_type == 2:
            self.correct_answer = f"#define {self.macro_name}(a, b) ((a) + (b))"
            desc = f"функциональный макрос {self.macro_name}(a, b), возвращающий сумму аргументов"
        else:
            self.correct_answer = f"#define {self.macro_name}(a, b) ((a) < (b) ? (a) : (b))"
            desc = f"функциональный макрос {self.macro_name}(a, b), возвращающий минимум из двух аргументов"
        
        self._cached_task_text = f"Напишите директиву #define, которая создаёт {desc}."
        self._task_generated = True  
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_define
        )]

    def _compare_define(self, output: str, expected: str) -> bool:
        out_norm = " ".join(output.strip().split())
        exp_norm = " ".join(expected.strip().split())
        return out_norm == exp_norm

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "#define" not in self.solution.lower():
            return "Ошибка: ответ должен содержать #define"
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


# Сложные задачи (2.3.х)


class Task2_3_1_IfndefDefine(BaseTaskClass):
    """2.3.1: Директивы: если макрос не определён — определить его"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.macro_name: str = f"MACRO{seed}"
        self.macro_val: int = 0
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False 

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.macro_val = self._rng.randint(1, 100)
        
        self.correct_answer = f"""#ifndef {self.macro_name}
#define {self.macro_name} {self.macro_val}
#endif"""
        
        self._cached_task_text = f"Напишите директивы препроцессора: если макрос {self.macro_name} не определён — определить его со значением {self.macro_val}"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_multiline_directive
        )]

    def _compare_multiline_directive(self, output: str, expected: str) -> bool:
        out_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        exp_lines = [line.strip() for line in expected.strip().split("\n") if line.strip()]
        return out_lines == exp_lines

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "#ifndef" not in self.solution.lower() or "#define" not in self.solution.lower():
            return "Ошибка: ответ должен содержать #ifndef и #define"
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )


class Task2_3_2_IfElifElse(BaseTaskClass):
    """2.3.2: Директивы с #if/#elif/#else для условного определения"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.macro_name: str = f"VALUE{seed}"
        self.result_name: str = f"RESULT{seed}"
        self.t1: int = self._rng.randint(50, 100)
        self.t2: int = self._rng.randint(10, 49)
        self.a: int = self._rng.randint(1, 10)
        self.b: int = self._rng.randint(1, 10)
        self.c: int = self._rng.randint(1, 10)
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.correct_answer = f"""#if {self.macro_name} > {self.t1}
#define {self.result_name} {self.a}
#elif {self.macro_name} > {self.t2}
#define {self.result_name} {self.b}
#else
#define {self.result_name} {self.c}
#endif"""
        
        return f"Напишите директивы препроцессора: если {self.macro_name} больше {self.t1} — определить макрос {self.result_name} как {self.a}, если больше {self.t2} — как {self.b}, иначе — как {self.c}."

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_multiline_directive
        )]

    def _compare_multiline_directive(self, output: str, expected: str) -> bool:
        out_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        exp_lines = [line.strip() for line in expected.strip().split("\n") if line.strip()]
        return out_lines == exp_lines

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "#if" not in self.solution.lower() or "#define" not in self.solution.lower():
            return "Ошибка: ответ должен содержать #if и #define"
        return None

    def compile(self) -> Optional[str]:
        return None

    def run_tests(self) -> tuple[bool, str]:
        if not self.tests:
            return False, "Тесты не сгенерированы"
        
        test = self.tests[0]
        passed = test.compare_func(self.solution, test.expected)
        
        if passed:
            return True, "OK"
        else:
            return False, self.make_failed_test_msg(
                test.showed_input,
                self.solution,
                test.expected
            )

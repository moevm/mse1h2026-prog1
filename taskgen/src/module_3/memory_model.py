from typing import Optional
import random
from pathlib import Path
import sys
import re

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from base_module.base_task import *
from constants import *

"""
память и ее модель
 - стек
 - куча
 - сегменты данных
 - bss
 - строковые литералы
"""

class Task1_1_1_SegmentByVar(BaseTaskClass):
    """1.1.1: В каком сегменте памяти хранится {переменная}?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.var_type: Optional[str] = None
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.var_type = self._rng.choice(list(SEGMENT_TO_VAR.keys()))
        self.correct_answer = self.var_type
        var_description = SEGMENT_TO_VAR[self.var_type]
        
        self._cached_task_text = f"В каком сегменте памяти хранится {var_description}?"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_segment
        )]

    def _compare_segment(self, output: str, expected: str) -> bool:
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

class Task1_2_1_AddressFormat(BaseTaskClass):
    """1_2_1: Вопросы про формат вывода адреса (%p, &, void*)"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.question: Optional[str] = None
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.question = self._rng.choice(list(ADDRESS_FORMAT_QUESTION))
        self.correct_answer = ADDRESS_FORMAT_ANSWER[self.question]
        
        self._cached_task_text = self.question
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_format
        )]

    def _compare_format(self, output: str, expected: str) -> bool:
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

class Task1_3_1_DataVsBSS(BaseTaskClass):
    """1_3_1: Чем отличается Data от BSS?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.correct_answer = "инициализация"
        
        self._cached_task_text = "Каким ключевым отличием сегмент Data отличается от BSS? (одним словом)"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_diff
        )]

    def _compare_diff(self, output: str, expected: str) -> bool:
        valid_answers = ["инициализация", "инициализирован", "init", "значение"]
        return output.strip().lower() in valid_answers

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

class Task1_4_1_AutomaticLifetime(BaseTaskClass):
    """1_4_1: Какое управление памятью характерно для переменных в стеке?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.correct_answer = "автоматическое"
        
        self._cached_task_text = "Какое управление памятью характерно для переменных в стеке? (одно слово)"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_direction
        )]

    def _compare_direction(self, output: str, expected: str) -> bool:
        valid_answers = ["вниз", "down", "к меньшим адресам"]
        return output.strip().lower() in valid_answers

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


class Task1_5_1_StaticLifetime(BaseTaskClass):
    """1_5_1: Сколько живёт статическая переменная?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.correct_answer = "всю программу"
        
        self._cached_task_text = "Сколько времени живёт статическая переменная? (одним словом)"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_lifetime
        )]

    def _compare_lifetime(self, output: str, expected: str) -> bool:
        valid_answers = ["всю программу", "всегда", "всю жизнь", "программы", "global"]
        return output.strip().lower() in valid_answers

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


class Task1_6_1_LocalScope(BaseTaskClass):
    """1.6.1: Где видна локальная переменная?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.correct_answer = "функция"
        
        self._cached_task_text = "В пределах чего видна локальная переменная? (одним словом)"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_scope
        )]

    def _compare_scope(self, output: str, expected: str) -> bool:
        valid_answers = ["функция", "функции", "блок", "scope", "локально"]
        return output.strip().lower() in valid_answers

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


class Task1_7_1_MemorySize(BaseTaskClass):
    """1.7.1: Какой сегмент обычно самый большой?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.correct_answer: Optional[str] = None
        self._task_generated: bool = False
        self._cached_task_text: str = ""

    def generate_task(self) -> str:
        if self._task_generated:
            return self._cached_task_text
        
        self.correct_answer = "Heap"
        
        self._cached_task_text = "Какой сегмент памяти обычно имеет наибольший размер? (Stack/Heap/Data/BSS)"
        self._task_generated = True
        
        return self._cached_task_text

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_size
        )]

    def _compare_size(self, output: str, expected: str) -> bool:
        valid_answers = ["heap", "куча"]
        return output.strip().lower() in valid_answers

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
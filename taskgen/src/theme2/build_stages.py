from typing import Optional, List, Tuple
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from base_module.base_task import *

from constants import *


# Простые задачи (1.1.x)


class Task1_1_1_FlagByStage(BaseTaskClass):
    """1.1.1: Какой флаг gcc останавливает сборку после этапа {этап}?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.stage: Optional[str] = None
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.stage = self._rng.choice(list(STAGE_TO_FLAG.keys()))
        self.correct_answer = STAGE_TO_FLAG[self.stage]
        return f"Какой флаг gcc останавливает сборку после этапа {self.stage}?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_flag
        )]

    def _compare_flag(self, output: str, expected: str) -> bool:
        return output.strip().upper() == expected.strip().upper()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if not self.solution.strip().startswith("-"):
            return "Ошибка: ожидается флаг вида -E, -S или -c"
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
    

class Task1_1_2_ExtByStage(BaseTaskClass):
    """1.1.2: Какое расширение получит файл после этапа {этап}?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.stage: Optional[str] = None
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.stage = self._rng.choice(list(STAGE_TO_EXT.keys()))
        self.correct_answer = STAGE_TO_EXT[self.stage]
        return f"Какое расширение получит файл после этапа {self.stage}?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_ext
        )]

    def _compare_ext(self, output: str, expected: str) -> bool:
        return output.strip().lower() == expected.strip().lower()

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if not self.solution.strip().startswith("."):
            return "Ошибка: ожидается расширение вида .?"
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


class Task1_1_3_StageByAction(BaseTaskClass):
    """1.1.3: На каком этапе сборки {действие}?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.action: Optional[str] = None
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        stage = self._rng.choice(list(STAGE_ACTIONS.keys()))
        self.action = self._rng.choice(STAGE_ACTIONS[stage])
        self.correct_answer = stage.capitalize()
        return f"На каком этапе сборки {self.action}?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_stage
        )]

    def _compare_stage(self, output: str, expected: str) -> bool:
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


class Task1_1_4_DefaultOutput(BaseTaskClass):
    """1.1.4: Как называется файл, создаваемый по умолчанию при полной сборке без -o?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.correct_answer = "a.out"

    def generate_task(self) -> str:
        return "Как называется файл, создаваемый по умолчанию при полной сборке без -o?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_default
        )]

    def _compare_default(self, output: str, expected: str) -> bool:
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


class Task1_1_5_ReadObjectFile(BaseTaskClass):
    """1.1.5: Можно ли прочитать объектный файл {name}.o в текстовом редакторе?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"file{seed}"
        self.correct_answer = "нет"

    def generate_task(self) -> str:
        return f"Можно ли прочитать объектный файл {self.file_name}.o в текстовом редакторе?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_yes_no
        )]

    def _compare_yes_no(self, output: str, expected: str) -> bool:
        return output.strip().lower() in ["нет", "no", "нельзя", "cannot"]

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


# Средние задачи (1.2.x)


class Task1_2_1_GccCommandByStage(BaseTaskClass):
    """1.2.1: Напишите команду gcc для выполнения только {этап}"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"prog{seed}"
        self.stage: Optional[str] = None
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.stage = self._rng.choice(list(STAGE_TO_FLAG.keys()))
        ext = STAGE_TO_EXT[self.stage]
        flag = STAGE_TO_FLAG[self.stage]
        self.correct_answer = f"gcc {flag} {self.file_name}.c -o {self.file_name}{ext}"
        return f"Дан файл {self.file_name}.c Напишите команду gcc, чтобы выполнить только {self.stage} и сохранить результат в {self.file_name}{ext}"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_command
        )]

    def _compare_command(self, output: str, expected: str) -> bool:
        out_parts = sorted(output.strip().split())
        exp_parts = sorted(expected.strip().split())
        return out_parts == exp_parts

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "gcc" not in self.solution.lower():
            return "Ошибка: команда должна содержать gcc"
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
        

class Task1_2_2_FullBuildCommand(BaseTaskClass):
    """1.2.2: Напишите одну команду gcc для полной сборки"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"source{seed}"
        self.prog_name: str = f"prog{seed}"
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.correct_answer = f"gcc {self.file_name}.c -o {self.prog_name}"
        return f"Дан файл {self.file_name}.c. Напишите одну команду gcc для полной сборки в исполняемый файл {self.prog_name}."

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_command
        )]

    def _compare_command(self, output: str, expected: str) -> bool:
        out_parts = sorted(output.strip().split())
        exp_parts = sorted(expected.strip().split())
        return out_parts == exp_parts

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "gcc" not in self.solution.lower():
            return "Ошибка: команда должна содержать gcc"
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
    

class Task1_2_3_OutputFileByFlag(BaseTaskClass):
    """1.2.3: Какой файл будет создан, если не указать -o?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"test{seed}"
        self.flag: Optional[str] = None
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        flags = ["-E", "-S", "-c", ""]
        self.flag = self._rng.choice(flags)
        
        if self.flag == "-E":
            self.correct_answer = "файл не создается"
        elif self.flag == "-S":
            self.correct_answer = f"{self.file_name}.s"
        elif self.flag == "-c":
            self.correct_answer = f"{self.file_name}.o"
        else:
            self.correct_answer = "a.out"
        
        flag_str = self.flag if self.flag else "(без флага)"
        return f"Дана команда gcc {flag_str} {self.file_name}.c. Какой файл будет создан, если не указать -o?"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_output_file
        )]

    def _compare_output_file(self, output: str, expected: str) -> bool:
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
    

class Task1_2_4_SeparateCompile(BaseTaskClass):
    """1.2.4: Команды для раздельной компиляции и линковки"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.n_files: int = self._rng.randint(2, 3)
        self.file_names: List[str] = [f"file{i}{seed}" for i in range(1, self.n_files + 1)]
        self.main_file: str = f"main{seed}"
        self.prog_name: str = f"prog{seed}"
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        files_str = ", ".join([f"{f}.c" for f in self.file_names])
        self.correct_answer = "\n".join([
            f"gcc -c {f}.c -o {f}.o" for f in self.file_names
        ]) + f"\ngcc {' '.join([f'{f}.o' for f in self.file_names])} -o {self.prog_name}"
        
        return f"Даны {self.n_files} файлов .c. Файл {self.main_file}.c вызывает функции из всех остальных. Напишите все команды для раздельной компиляции каждого файла и финальной линковки в {self.prog_name}."

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_multiline_command
        )]

    def _compare_multiline_command(self, output: str, expected: str) -> bool:
        out_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        exp_lines = [line.strip() for line in expected.strip().split("\n") if line.strip()]
        
        compile_cmds_out = [l for l in out_lines if l.startswith("gcc -c")]
        compile_cmds_exp = [l for l in exp_lines if l.startswith("gcc -c")]

        link_cmd_out = [l for l in out_lines if not l.startswith("gcc -c") and "gcc" in l]
        link_cmd_exp = [l for l in exp_lines if not l.startswith("gcc -c") and "gcc" in l]
        
        return (sorted(compile_cmds_out) == sorted(compile_cmds_exp) and 
                len(link_cmd_out) == len(link_cmd_exp))

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "gcc" not in self.solution.lower():
            return "Ошибка: команды должны содержать gcc"
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
    

class Task1_2_5_ErrorStageChoice(BaseTaskClass):
    """1.2.5: На каком этапе сборки возникнет ошибка? И какая?"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.pattern_idx: int = seed % len(ERROR_PATTERNS)
        self.code: Optional[str] = None
        self.correct_stage: Optional[str] = None
        self.correct_error: Optional[str] = None

    def generate_task(self) -> str:
        pattern = ERROR_PATTERNS[self.pattern_idx]
        
        self.code = pattern["code"].format(
            fake=f"fake{self._rng.randint(1, 100)}",
            var=f"var{self._rng.randint(1, 100)}",
            val=self._rng.randint(1, 100),
            func=f"func{self._rng.randint(1, 100)}",
            k=self._rng.randint(1, 10),
        )
        
        self.correct_stage = pattern["stage"]
        self.correct_error = pattern["error"]
        
        return f"Дан файл test.c:\n```c\n{self.code}\n```\nНа каком этапе сборки возникнет ошибка? И какая?\n\nВарианты этапа:\na) Препроцессинг\nb) Компиляция\nc) Ассемблирование\nd) Линковка\n\nВарианты ошибки:\na) Файл не найден\nb) Синтаксическая ошибка\nc) Неразрешённый символ\nd) Повторное определение"

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=f"{self.correct_stage} / {self.correct_error}",
            compare_func=self._compare_error_choice
        )]

    def _compare_error_choice(self, output: str, expected: str) -> bool:
        out_lower = output.lower()
        stage_match = any(s in out_lower for s in [self.correct_stage.lower(), self.correct_stage[0].lower()])
        error_match = any(e in out_lower for e in [self.correct_error.lower(), self.correct_error[0].lower()])
        return stage_match and error_match

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
    

# Сложные задачи (1.3.x)


class Task1_3_1_LinkObjectFiles(BaseTaskClass):
    """1.3.1: Команда gcc для линковки всех объектных файлов"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.n_files: int = self._rng.randint(2, 4)
        self.file_names: List[str] = [f"obj{i}{seed}" for i in range(1, self.n_files + 1)]
        self.prog_name: str = f"prog{seed}"
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        files_str = ", ".join([f"{f}.o" for f in self.file_names])
        self.correct_answer = f"gcc {' '.join([f'{f}.o' for f in self.file_names])} -o {self.prog_name}"
        return f"Даны {self.n_files} файлов: {files_str}. Все уже скомпилированы в .o. Напишите одну команду gcc для линковки всех объектных файлов в исполняемый файл {self.prog_name}."

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_command
        )]

    def _compare_command(self, output: str, expected: str) -> bool:
        out_parts = sorted(output.strip().split())
        exp_parts = sorted(expected.strip().split())
        return out_parts == exp_parts

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "gcc" not in self.solution.lower():
            return "Ошибка: команда должна содержать gcc"
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


class Task1_3_2_ContinueBuild(BaseTaskClass):
    """1.3.2: Команда gcc чтобы продолжить сборку с промежуточного файла"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_name: str = f"prog{seed}"
        self.prog_name: str = f"final{seed}"
        self.flag: Optional[str] = None
        self.intermediate_file: Optional[str] = None
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        flags_files = [
            ("-E", f"{self.file_name}.i"),
            ("-S", f"{self.file_name}.s"),
            ("-c", f"{self.file_name}.o"),
        ]
        self.flag, self.intermediate_file = self._rng.choice(flags_files)
        self.correct_answer = f"gcc {self.intermediate_file} -o {self.prog_name}"
        
        return f"Дан файл {self.file_name}.c. Студент выполнил команду gcc {self.flag} {self.file_name}.c и получил файл {self.intermediate_file}. Напишите команду gcc, чтобы продолжить сборку с этого места и получить исполняемый файл {self.prog_name}."

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_command
        )]

    def _compare_command(self, output: str, expected: str) -> bool:
        out_parts = sorted(output.strip().split())
        exp_parts = sorted(expected.strip().split())
        return out_parts == exp_parts

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "gcc" not in self.solution.lower():
            return "Ошибка: команда должна содержать gcc"
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


class Task1_3_3_TwoFileBuild(BaseTaskClass):
    """1.3.3: 3 команды для раздельной сборки и линковки двух файлов"""
    
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)
        self.file_a: str = f"module{seed}"
        self.file_b: str = f"utils{seed}"
        self.func_name: str = f"helper{seed}"
        self.prog_name: str = f"app{seed}"
        self.correct_answer: Optional[str] = None

    def generate_task(self) -> str:
        self.correct_answer = "\n".join([
            f"gcc -c {self.file_a}.c -o {self.file_a}.o",
            f"gcc -c {self.file_b}.c -o {self.file_b}.o",
            f"gcc {self.file_a}.o {self.file_b}.o -o {self.prog_name}",
        ])
        return f"Даны файлы {self.file_a}.c и {self.file_b}.c. Файл {self.file_a}.c вызывает {self.func_name}(), определённую в {self.file_b}.c. Напишите 3 команды для раздельной сборки и линковки в {self.prog_name}."

    def _generate_tests(self):
        self.tests = [TestItem(
            input_str=self.solution,
            showed_input=self.solution if self.solution else "(пусто)",
            expected=self.correct_answer,
            compare_func=self._compare_multiline_command
        )]

    def _compare_multiline_command(self, output: str, expected: str) -> bool:
        out_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        exp_lines = [line.strip() for line in expected.strip().split("\n") if line.strip()]
        return sorted(out_lines) == sorted(exp_lines)

    def check_sol_prereq(self) -> Optional[str]:
        if not self.solution or not self.solution.strip():
            return "Ошибка: ответ не может быть пустым."
        if "gcc" not in self.solution.lower():
            return "Ошибка: команды должны содержать gcc"
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
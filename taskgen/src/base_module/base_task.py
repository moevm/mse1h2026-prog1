from typing import Optional, Callable
from dataclasses import dataclass
import os
import subprocess
import shlex

DEFAULT_TEST_NUM = 50
COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 5


@dataclass
class TestItem:
    # структура одного теста: вход, отображаемый вход, ожидаемый вывод, функция сравнения
    input_str: str
    showed_input: str
    expected: str
    compare_func: Callable[[str, str], bool]


class BaseTaskClass:
    def __init__(
        self,
        compile_name: str = "prog.x",
        seed: int = 0,
        tests_num: int = DEFAULT_TEST_NUM,
        fail_on_first_test: bool = True,
        array_align: str = "center",
        jail_exec: str = "chroot",
        jail_path: Optional[str] = None,
    ):
        # инициализация базовых параметров задачи
        self.solution = ""
        self.check_files: dict[str, str] = {}
        self.static_check_files: dict[str, str] = {}
        self.prog_name = compile_name
        self.seed = seed
        self.tests_num = tests_num
        self.tests: list[TestItem] = []
        self.compile_timeout = COMPILE_TIMEOUT
        self.run_timeout = RUN_TIMEOUT
        self.fail_on_first = fail_on_first_test
        self._array_align = array_align
        self.allowed_symbols: list[str] = []
        self.jail_exec = jail_exec
        self.jail_path = jail_path if jail_path is not None else os.environ.get("JAIL_PATH", "")

    def load_student_solution(self, solfile: Optional[str] = None, solcode: Optional[str] = None):
        # загрузка решения студента (из строки или файла)
        if solcode is None and solfile is None:
            raise ValueError("Не указан код решения или путь к файлу.")
        if solcode is not None and solfile is not None:
            raise ValueError("Переданы и код, и файл одновременно.")
        if solcode is not None:
            self.solution = solcode.strip()
        else:
            if not os.path.exists(solfile):
                raise ValueError("Файл решения не найден.")
            with open(solfile, "r", encoding="utf-8") as f:
                self.solution = f.read().strip()

    def check_sol_prereq(self) -> Optional[str]:
        # базовая проверка решения перед компиляцией (можно расширять в наследниках)
        if not self.solution or self.solution.strip() == "":
            return "Ошибка: пустой файл."
        return None

    @staticmethod
    def _dump_files(files: dict[str, str]):
        # запись файлов системы проверки во временную директорию
        for name, content in files.items():
            with open(name, "w", encoding="utf-8") as f:
                f.write(content)

    def _compile_file(self, file: str, compiler: str, compile_args: str, output: str = None):
        # компиляция одного файла
        if output is not None:
            compile_args = f"{compile_args} -o {output}"
        compile_command = f"{compiler} -c {compile_args} {file}"
        p = subprocess.run(
            shlex.split(compile_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.compile_timeout,
            check=False
        )
        if p.returncode != 0:
            return p.stdout.decode()
        return None

    def compile_static(self) -> Optional[str]:
        # компиляция статических файлов проверки (можно переопределять в наследниках)
        self.static_check_files = {
            self.__class__.__name__ + "_" + name: content
            for name, content in self.static_check_files.items()
        }
        self._dump_files(self.static_check_files)
        for src_file in self.static_check_files.keys():
            err = self._compile_file(src_file, "gcc", "-static")
            if err is not None:
                return f"Ошибка компиляции статического файла {src_file}:\n{err}"
            os.remove(src_file)
        return None

    def _compile_internal(
        self,
        solution_name: str = "solution.c",
        compiler: str = "gcc",
        compile_args: str = "-O2",
        keep_static_obj_files: bool = True
    ) -> Optional[str]:
        self._dump_files(self.check_files)

        with open(solution_name, "w", encoding="utf-8") as f:
            f.write(self.solution + "\n")

        obj_files = []
        for src_file in self.check_files.keys():
            err = self._compile_file(src_file, compiler, compile_args)
            if err is not None:
                return f"Ошибка компиляции проверяющего кода ({src_file}):\n{err}"
            obj_files.append(src_file[:src_file.rfind('.')] + ".o")

        stud_obj = "stud_work.o"
        err = self._compile_file(solution_name, compiler, compile_args, stud_obj)
        if err is not None:
            return f"Ошибка компиляции решения:\n{err}"
        obj_files.append(stud_obj)

        exe_path = os.path.join(self.jail_path, self.prog_name)
        compile_command = [compiler, *obj_files] + compile_args.split() + ["-o", exe_path]
        p = subprocess.run(
            compile_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False
        )
        if p.returncode != 0:
            return f"Ошибка линковки:\n{p.stdout.decode()}"
        return None

    def compile(self) -> Optional[str]:
        # основной метод компиляции (можно переопределять в наследниках)
        return self._compile_internal()

    def generate_task(self) -> str:
        # генерация условия задачи (переопределить в наследнике)
        return "TODO: BaseTaskClass"

    def _generate_tests(self):
        # генерация тестов (переопределить в наследнике)
        pass

    def _run_solution_internal(
        self, test: TestItem,
        emulator: str = "", emu_args: str = "", prog_args: str = ""
    ) -> Optional[tuple[str, str]]:
        # Внутренний запуск программы на одном тесте
        exe_path = os.path.join(self.jail_path, self.prog_name)

        run_cmd = [self.jail_exec] if self.jail_path else []
        if self.jail_path:
            run_cmd += [self.jail_path]
        run_cmd.append(exe_path)

        try:
            p = subprocess.run(
                run_cmd + shlex.split(prog_args),
                input=test.input_str.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.run_timeout,
                check=False
            )
        except subprocess.TimeoutExpired as e:
            return (
                f"Превышено время выполнения ({e.timeout} сек).",
                f"Ожидалось завершение за {e.timeout} сек."
            )

        output = p.stdout.decode().strip() + p.stderr.decode().strip()

        if p.returncode != 0:
            return (
                f"Программа завершилась с кодом {p.returncode}.\n{output}",
                "Код возврата 0"
            )

        passed = test.compare_func(output, test.expected)
        if passed:
            return None

        return output, test.expected

    def _compare_default(self, output: str, expected: str) -> bool:
        # стандартное сравнение вывода
        return output.strip() == expected.strip()

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        # публичный запуск решения на одном тесте
        return self._run_solution_internal(test)

    def make_failed_test_msg(self, showed_input: str, obtained: str, expected: str) -> str:
        # формирование сообщения об ошибке теста
        return (
            f"Тест не пройден.\n"
            f"Вход: '{showed_input}'\n"
            f"Получено: '{obtained}'\n"
            f"Ожидалось: '{expected}'"
        )

    def run_tests(self) -> tuple[bool, str]:
        # последовательный запуск всех тестов
        msgs: list[str] = []

        for t in self.tests:
            result = self.run_solution(t)
            if result is not None:
                msgs.append(self.make_failed_test_msg(
                    t.showed_input,
                    result[0],
                    result[1]
                ))
                if self.fail_on_first:
                    break

        if not msgs:
            return True, "OK"

        return False, "\n".join(msgs)

    def init_task(self) -> str:
        # инициализация задачи (вызывается при создании задания)
        return self.generate_task()

    def check(self) -> tuple[bool, str]:
        # полный цикл проверки решения
        try:
            err = self.check_sol_prereq()
            if err:
                return False, err

            err = self.compile()
            if err:
                return False, err

            self.generate_task()
            self._generate_tests()

            return self.run_tests()

        except Exception as e:
            return False, f"Непредвиденная ошибка: {str(e)}"

    def make_array_failed_test_msg(
        self,
        caption: list[str],
        arrs: list[list],
        max_col_len: int,
        correctness: list[bool]
    ) -> str:
        # формирование таблицы сравнения массивов
        ret = ""
        cols = ["i"]
        cols_lens = [max(len(cols[0]), len(str(len(correctness))))]

        cols += caption
        cols_lens += [max(max_col_len, len(col)) for col in cols[1:]]

        cols.append("Correct")
        correct_s, fail_s = "V", "X"
        cols_lens.append(max(map(len, (correct_s, fail_s, cols[-1]))))

        ret += "| " + " | ".join(
            col.center(cols_lens[i]) for i, col in enumerate(cols)
        ) + " |\n"

        separator = "+" + "+".join("-" * (col_len + 2) for col_len in cols_lens) + "+\n"
        ret += separator

        corr_iter = (correct_s if c else fail_s for c in correctness)

        for i, *vals in enumerate(zip(*arrs, correctness)):
            row = [str(i)] + [str(v) for v in vals[:-1]] + [next(corr_iter)]
            ret += "| " + " | ".join(
                self._align_value(val, cols_lens[j])
                for j, val in enumerate(row)
            ) + " |\n"

        ret += separator
        return ret

    def _align_value(self, value: str, max_len: int) -> str:
        # выравнивание значения в таблице
        if self._array_align == "center":
            return str(value).center(max_len)
        if self._array_align == "left":
            return str(value).ljust(max_len)
        if self._array_align == "right":
            return str(value).rjust(max_len)
        return str(value)
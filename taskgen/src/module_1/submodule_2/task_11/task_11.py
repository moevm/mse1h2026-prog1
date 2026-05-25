from typing import Optional
from src.base_module.base_task import BaseTaskClass, TestItem


def generate_word1(seed: int) -> str:
    words = ["Hello", "Test", "Foo", "Bar", "42", "abc", "xyz", "Hi"]
    return words[seed % len(words)]


def generate_word2(seed: int) -> str:
    words = ["world", "result", "baz", "qux", "100", "end", "start", "value"]
    return words[seed % len(words)]


def generate_symbol(seed: int):
    symbols = [
        ('\n', '\\n'),
        ('\t', '\\t'),
        ('\\', '\\\\'),
        ('"', '\\"')
    ]
    return symbols[seed % len(symbols)]


class Module_1_Submodule_2_task_11(BaseTaskClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word1 = generate_word1(self.seed)
        self.word2 = generate_word2(self.seed // 8)
        self.symbol_char, self.symbol_esc = generate_symbol(self.seed // 64)
        self.text = self.word1 + self.symbol_char + self.word2
        self.expected_string = self.word1 + self.symbol_esc + self.word2
        self.student_solution = ""

    def generate_task(self) -> str:
        code = (
            '#include <stdio.h>\n\n'
            'int main() {\n'
            '    printf("Ваша строка");\n'
            '    return 0;\n'
            '}'
        )
        return (
            f"Дана программа, которая выводит на экран строку:\n"
            f"{self.text}\n"
            f"Программа:\n"
            f"```\n{code}\n```\n"
            f"Определите, что должно быть на месте `Ваша строка`, чтобы программа выводила строку, представленную выше. "
            f"Используйте эскейп последовательности."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=self.expected_string,
                compare_func=lambda output, exp: output.strip() == exp.strip()
            )
        ]

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
            expected = self.expected_string
            student = self.student_solution.strip()
            if student.startswith('"') and student.endswith('"'):
                student = student[1:-1]
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
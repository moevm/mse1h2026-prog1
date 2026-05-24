from dataclasses import dataclass
from typing import Optional
import re
import itertools

from src.base_module.base_task import BaseTaskClass, TestItem


FIXED_OUTPUTS = [
    "error: No such file or directory",
    "error: Permission denied",
    "error: Invalid argument",
    "error: Numerical result out of range",
]

FRAGMENT_TO_INDEX = {
    1: 0,  # ENOENT
    2: 1,  # EACCES
    3: 2,  # EINVAL
    4: 3,  # ERANGE
}


class PerrorTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.permutation = self._get_permutation(seed_value % 24)
        self.label_to_output = {
            'A': FIXED_OUTPUTS[self.permutation[0]],
            'B': FIXED_OUTPUTS[self.permutation[1]],
            'C': FIXED_OUTPUTS[self.permutation[2]],
            'D': FIXED_OUTPUTS[self.permutation[3]],
        }
        self.correct_answer = []
        for frag in range(1, 5):
            needed_output = FIXED_OUTPUTS[FRAGMENT_TO_INDEX[frag]]
            label = next(label for label, out in self.label_to_output.items() if out == needed_output)
            self.correct_answer.append(label)
        self.correct_answer_str = ''.join(self.correct_answer)

    def _get_permutation(self, n: int) -> list[int]:
        """Возвращает перестановку [0,1,2,3] для n от 0 до 23."""
        perms = list(itertools.permutations([0, 1, 2, 3]))
        return list(perms[n])

    def generate_task(self) -> str:
        fragments = []
        for i in range(1, 5):
            errno_names = ["ENOENT", "EACCES", "EINVAL", "ERANGE"]
            fragments.append(f"**Фрагмент {i}:**\n```c\nerrno = {errno_names[i-1]};\nperror(\"error\");\n```")
        fragments_text = "\n\n".join(fragments)

        outputs_text = "\n".join([f"| {label} | `{out}` |" for label, out in self.label_to_output.items()])
        table = f"| Метка | Строка вывода |\n|-------|---------------|\n{outputs_text}"

        return (
            "# Обработка ошибок: perror\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            "  Ниже приведены 4 фрагмента кода (1–4) и 4 строки вывода (A–D).\n"
            "  Сопоставьте каждый фрагмент с тем выводом, который он произведёт в `stderr`.\n"
            "  Введите ответ в виде 4 букв без пробелов — i-я буква соответствует фрагменту i.\n"
            "  Например: `BCAD` означает: фрагмент 1 → B, фрагмент 2 → C, фрагмент 3 → A, фрагмент 4 → D.\n\n"
            f"**Фрагменты кода:**\n\n{fragments_text}\n\n"
            f"**Варианты вывода (A–D):**\n\n{table}\n\n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="[скрыто]",
                expected=self.correct_answer_str,
                compare_func=self._compare_default,
            )
        ]

    def _check_solution_text(self) -> bool:
        answer = self.solution.strip().upper()
        answer = re.sub(r'[^A-D]', '', answer)
        if len(answer) != 4:
            return False
        return answer == self.correct_answer_str

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        if self._check_solution_text():
            return None
        return self.solution.strip(), "[скрыто]"
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class VariantSpec:
    fields_set: str
    orderings: tuple[tuple[str, ...], ...]


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        fields_set="char, double, char, int",
        orderings=(
            ("double a", "int b",    "char c",  "char d"),
            ("char a",   "double b", "int c",   "char d"),
            ("char a",   "char b",   "int c",   "double d"),
            ("int a",    "char b",   "double c","char d"),
        ),
    ),
    1: VariantSpec(
        fields_set="char, int, short, double",
        orderings=(
            ("double a", "int b",   "short c", "char d"),
            ("char a",   "int b",   "short c", "double d"),
            ("char a",   "short b", "int c",   "double d"),
            ("int a",    "char b",  "double c","short d"),
        ),
    ),
    2: VariantSpec(
        fields_set="int, char, short, long long",
        orderings=(
            ("long long a", "int b",  "short c", "char d"),
            ("int a",       "char b", "short c", "long long d"),
            ("char a",      "int b",  "long long c", "short d"),
            ("short a",     "int b",  "char c",  "long long d"),
        ),
    ),
    3: VariantSpec(
        fields_set="char, int, char, short, double",
        orderings=(
            ("double a", "int b",  "short c", "char d", "char e"),
            ("char a",   "int b",  "char c",  "short d","double e"),
            ("char a",   "char b", "int c",   "double d","short e"),
            ("int a",    "char b", "short c", "double d","char e"),
        ),
    ),
}

_LABELS = ("A", "B", "C", "D")


def _fields_decl(ordering: tuple[str, ...]) -> str:
    return "\n    ".join(f"{f};" for f in ordering)


class StructOptimalTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

        # Перемешиваем метки на основе seed: правильный вариант — orderings[0]
        # Генерируем перестановку меток, зависящую от seed
        import random
        rng = random.Random(seed_value)
        label_order = list(_LABELS)
        rng.shuffle(label_order)
        # label_order[i] — буква, которая будет показана для orderings[i]
        self._label_for_ordering = label_order
        # Правильный вариант — orderings[0], его буква:
        self._correct_label = label_order[0]

    def _get_sizeof(self, ordering: tuple[str, ...]) -> Optional[int]:
        decl = _fields_decl(ordering)
        source = textwrap.dedent(f"""\
            #include <stdio.h>

            struct S {{
                {decl}
            }};

            int main(void) {{
                printf("%zu\\n", sizeof(struct S));
                return 0;
            }}
        """)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "sizeof.c"
            exe_path = tmp_path / "sizeof.x"
            src_path.write_text(source, encoding="utf-8")
            cp = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if cp.returncode != 0:
                return None
            rp = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, check=False,
            )
            try:
                return int(rp.stdout.decode().strip())
            except ValueError:
                return None

    def generate_task(self) -> str:
        v = self.variant
        lines = []
        for i, ordering in enumerate(v.orderings):
            label = self._label_for_ordering[i]
            fields_str = "; ".join(ordering)
            lines.append((label, f"  **{label}.** `struct S {{ {fields_str}; }}`"))

        lines.sort(key=lambda x: x[0])
        variants_block = "\n".join(text for _, text in lines)

        return (
            "# Struct: оптимальный порядок полей\n\n"
            "### Задание №6\n\n"
            "- **Формулировка:**  \n"
            f"  Набор полей: `{v.fields_set}`  \n\n"
            "  Ниже представлены четыре варианта объявления — порядок полей разный.  \n"
            "  Определите, какой вариант даёт **наименьший** `sizeof(struct S)` "
            "на платформе x86-64.  \n\n"
            f"{variants_block}\n\n"
            "  Введите ответ в формате: `<буква> <sizeof>`  \n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        sizeof_correct = self._get_sizeof(self.variant.orderings[0])
        expected = f"{self._correct_label} {sizeof_correct}" if sizeof_correct else "[скрыто]"
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"набор полей: {self.variant.fields_set}",
                expected=expected,
                compare_func=self._compare_default,
            )
        ]

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        answer = self.solution.strip()
        parts = answer.split()

        if len(parts) != 2:
            return "Неверно", "[скрыто]"

        letter, sizeof_str = parts[0].upper(), parts[1]

        if letter not in _LABELS:
            return "Неверно", "[скрыто]"

        try:
            student_sizeof = int(sizeof_str)
        except ValueError:
            return "Неверно", "[скрыто]"

        correct_sizeof = self._get_sizeof(self.variant.orderings[0])
        if correct_sizeof is None:
            return "Ошибка при вычислении sizeof", "[скрыто]"

        if letter != self._correct_label or student_sizeof != correct_sizeof:
            return "Неверно", "[скрыто]"

        return None
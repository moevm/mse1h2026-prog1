from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import textwrap

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class MemberSpec:
    name: str
    value: int


@dataclass(frozen=True)
class VariantSpec:
    enum_name: str
    members: tuple[MemberSpec, ...] 
    signature: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        enum_name="Direction",
        members=(
            MemberSpec("NORTH", 0),
            MemberSpec("EAST",  1),
            MemberSpec("SOUTH", 2),
            MemberSpec("WEST",  3),
        ),
        signature="enum Direction next_Direction(enum Direction d)",
    ),
    1: VariantSpec(
        enum_name="Season",
        members=(
            MemberSpec("SPRING", 0),
            MemberSpec("SUMMER", 1),
            MemberSpec("AUTUMN", 2),
            MemberSpec("WINTER", 3),
        ),
        signature="enum Season next_Season(enum Season d)",
    ),
    2: VariantSpec(
        enum_name="Suit",
        members=(
            MemberSpec("CLUBS",    0),
            MemberSpec("DIAMONDS", 1),
            MemberSpec("HEARTS",   2),
            MemberSpec("SPADES",   3),
        ),
        signature="enum Suit next_Suit(enum Suit d)",
    ),
    3: VariantSpec(
        enum_name="Channel",
        members=(
            MemberSpec("RED",   0),
            MemberSpec("GREEN", 1),
            MemberSpec("BLUE",  2),
            MemberSpec("ALPHA", 3),
        ),
        signature="enum Channel next_Channel(enum Channel d)",
    ),
}

_COUNT = 4


class EnumNextTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        return (
            "# Enum: значения по умолчанию\n\n"
            "### Задание №1\n\n"
            "- **Формулировка:**  \n"
            f"  Напишите функцию `next_{v.enum_name}`, которая:  \n"
            f"  1. Принимает значение типа `enum {v.enum_name}`.  \n"
            "  2. Возвращает **следующий** элемент перечисления по кругу "
            "(после последнего идёт первый).  \n\n"
            f"  Сигнатура: `{v.signature}`  \n\n"
            "  Объявление `enum` и `main` писать не нужно — они предоставляются системой.  \n"
            "  Подключать заголовочные файлы не требуется.  \n"
            f"  Количество элементов в перечислении: **{_COUNT}**.  \n"
            "  Конкретные имена членов не известны — "
            "опирайтесь только на числовые значения."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = [
            TestItem(
                input_str="",
                showed_input=f"next_{v.enum_name}({m.value})",
                expected=str(v.members[(i + 1) % _COUNT].value),
                compare_func=self._compare_default,
            )
            for i, m in enumerate(v.members)
        ]

    def _build_enum_decl(self) -> str:
        v = self.variant
        members_str = ",\n    ".join(
            f"{m.name} = {m.value}" for m in v.members
        )
        return f"enum {v.enum_name} {{\n    {members_str}\n}};"

    def _build_program_source(self, input_value: int) -> str:
        v = self.variant
        enum_decl = self._build_enum_decl()
        return textwrap.dedent(f"""\
            #include <stdio.h>

            {enum_decl}

            {self.solution}

            int main(void) {{
                enum {v.enum_name} result = next_{v.enum_name}((enum {v.enum_name}){input_value});
                printf("%d\\n", (int)result);
                return 0;
            }}
        """)

    def _compile_and_run(self, input_value: int) -> tuple[bool, str]:
        program_source = self._build_program_source(input_value)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            compile_proc = subprocess.run(
                [
                    "gcc", "-std=c11", "-O2",
                    "-Werror=int-conversion",  # return int вместо enum → ошибка
                    str(src_path), "-o", str(exe_path),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stdout.decode()

            run_proc = subprocess.run(
                [str(exe_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            output = "\n".join(
                part for part in (
                    run_proc.stdout.decode().strip(),
                    run_proc.stderr.decode().strip(),
                ) if part
            )
            if run_proc.returncode != 0:
                return False, output
            return True, output

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        test_index = self.tests.index(test)
        input_value = self.variant.members[test_index].value

        ok, result = self._compile_and_run(input_value)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
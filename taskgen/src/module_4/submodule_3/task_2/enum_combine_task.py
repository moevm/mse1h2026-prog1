from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
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
    op_description: str 
    op_func: Callable[[int, int], int]
    op_c_expr: str    
    signature: str


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec(
        enum_name="HttpStatus",
        members=(
            MemberSpec("OK",           200),
            MemberSpec("FORBIDDEN",    403),
            MemberSpec("NOT_FOUND",    404),
            MemberSpec("SERVER_ERROR", 500),
        ),
        op_description="сумма двух значений (`a + b`)",
        op_func=lambda a, b: a + b,
        op_c_expr="(int)a + (int)b",
        signature="int combine_HttpStatus(enum HttpStatus a, enum HttpStatus b)",
    ),
    1: VariantSpec(
        enum_name="Priority",
        members=(
            MemberSpec("LOW",      1),
            MemberSpec("MEDIUM",   5),
            MemberSpec("HIGH",     10),
            MemberSpec("CRITICAL", 100),
        ),
        op_description="максимум из двух значений",
        op_func=lambda a, b: max(a, b),
        op_c_expr="((int)a > (int)b ? (int)a : (int)b)",
        signature="int combine_Priority(enum Priority a, enum Priority b)",
    ),
    2: VariantSpec(
        enum_name="Permission",
        members=(
            MemberSpec("NONE",  0),
            MemberSpec("EXEC",  1),
            MemberSpec("WRITE", 2),
            MemberSpec("READ",  4),
        ),
        op_description="побитовое ИЛИ двух значений (`a | b`)",
        op_func=lambda a, b: a | b,
        op_c_expr="(int)a | (int)b",
        signature="int combine_Permission(enum Permission a, enum Permission b)",
    ),
    3: VariantSpec(
        enum_name="LogLevel",
        members=(
            MemberSpec("DEBUG", 10),
            MemberSpec("INFO",  20),
            MemberSpec("WARN",  30),
            MemberSpec("ERROR", 40),
        ),
        op_description="разность двух значений (`a - b`)",
        op_func=lambda a, b: a - b,
        op_c_expr="(int)a - (int)b",
        signature="int combine_LogLevel(enum LogLevel a, enum LogLevel b)",
    ),
}


def _make_test_pairs(members: tuple[MemberSpec, ...]) -> list[tuple[MemberSpec, MemberSpec]]:
    n = len(members)
    pairs = []
    for i in range(n):
        for j in range(i, n):
            pairs.append((members[i], members[j]))
    return pairs


class EnumCombineTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]
        self._test_pairs = _make_test_pairs(self.variant.members)

    def generate_task(self) -> str:
        v = self.variant
        members_str = ", ".join(f"`{m.name}={m.value}`" for m in v.members)
        return (
            "# Enum: явное задание значений\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            f"  Объявите перечисление `{v.enum_name}` с явно заданными значениями:  \n"
            f"  {members_str}  \n\n"
            f"  Напишите функцию `combine_{v.enum_name}`, которая:  \n"
            f"  1. Принимает два значения типа `enum {v.enum_name}`.  \n"
            f"  2. Возвращает **{v.op_description}** как `int`.  \n\n"
            f"  Сигнатура: `{v.signature}`  \n\n"
            "  Писать `main` не нужно — только объявление `enum` и тело функции."
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        v = self.variant
        self.tests = [
            TestItem(
                input_str="",
                showed_input=(
                    f"combine_{v.enum_name}"
                    f"({ma.name}, {mb.name})"
                ),
                expected=str(v.op_func(ma.value, mb.value)),
                compare_func=self._compare_default,
            )
            for ma, mb in self._test_pairs
        ]

    def _build_enum_decl(self) -> str:
        v = self.variant
        members_str = ",\n    ".join(f"{m.name} = {m.value}" for m in v.members)
        return f"enum {v.enum_name} {{\n    {members_str}\n}};"

    def _build_program_source(self, name_a: str, name_b: str) -> str:
        v = self.variant
        return textwrap.dedent(f"""\
            #include <stdio.h>

            {self.solution}

            int main(void) {{
                int result = combine_{v.enum_name}({name_a}, {name_b});
                printf("%d\\n", result);
                return 0;
            }}
        """)
    

    def _compile_and_run(self, name_a: str, name_b: str) -> tuple[bool, str]:
        program_source = self._build_program_source(name_a, name_b)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_path = tmp_path / "check_program.c"
            exe_path = tmp_path / "check_program.x"

            src_path.write_text(program_source, encoding="utf-8")
            compile_proc = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src_path), "-o", str(exe_path)],
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
        ma, mb = self._test_pairs[test_index]

        ok, result = self._compile_and_run(ma.name, mb.name)
        if ok:
            if self._compare_default(result, test.expected):
                return None
            return result, test.expected
        return result, test.expected
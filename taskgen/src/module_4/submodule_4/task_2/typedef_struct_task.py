from dataclasses import dataclass
from typing import Optional
import re

from src.base_module.base_task import BaseTaskClass, TestItem


@dataclass(frozen=True)
class FieldSpec:
    c_type: str  
    name: str 


@dataclass(frozen=True)
class VariantSpec:
    type_name: str
    fields: tuple[FieldSpec, ...]


_VARIANTS: dict[int, VariantSpec] = {
    0: VariantSpec("Student", (
        FieldSpec("char[32]", "name"),
        FieldSpec("int",      "age"),
        FieldSpec("float",    "gpa"),
    )),
    1: VariantSpec("Book", (
        FieldSpec("char[64]", "title"),
        FieldSpec("int",      "pages"),
        FieldSpec("double",   "price"),
    )),
    2: VariantSpec("Point2D", (
        FieldSpec("double", "x"),
        FieldSpec("double", "y"),
        FieldSpec("int",    "label"),
    )),
    3: VariantSpec("Rectangle", (
        FieldSpec("double",  "width"),
        FieldSpec("double",  "height"),
        FieldSpec("char[16]","color"),
    )),
    4: VariantSpec("Employee", (
        FieldSpec("char[32]", "name"),
        FieldSpec("int",      "id"),
        FieldSpec("double",   "salary"),
    )),
}


def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def _parse_typedef_struct(solution: str, variant: VariantSpec) -> tuple[bool, str]:
    text = _normalize(solution.strip().rstrip(';'))

    if not re.match(r'^typedef\s+struct\b', text, re.IGNORECASE):
        return False, "Неверно"

    m = re.search(r'\{(.+)\}\s*(\w+)$', text, re.DOTALL)
    if not m:
        return False, "Неверно"

    body_raw = m.group(1)
    alias = m.group(2)

    if alias != variant.type_name:
        return False, f"Неверно"

    field_tokens = [_normalize(f) for f in body_raw.split(';') if _normalize(f)]

    if len(field_tokens) != len(variant.fields):
        return False, (
            f"Ожидается {len(variant.fields)} поля, "
            f"найдено {len(field_tokens)}: {field_tokens}"
        )

    for i, (token, expected) in enumerate(zip(field_tokens, variant.fields)):
        if '[' in expected.c_type:
            base, size = expected.c_type.split('[')
            size = size.rstrip(']')
            expected_field = _normalize(f"{base} {expected.name}[{size}]")
        else:
            expected_field = _normalize(f"{expected.c_type} {expected.name}")

        token_norm = _normalize(token)
        if token_norm != expected_field:
            return False, (
                f"Неверно"
            )

    return True, "ok"


class TypedefStructTask(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed_value = 0 if self.seed is None else self.seed
        self.variant_index = seed_value % len(_VARIANTS)
        self.variant = _VARIANTS[self.variant_index]

    def generate_task(self) -> str:
        v = self.variant
        fields_str = ", ".join(
            f"`{f.c_type} {f.name}`" for f in v.fields
        )
        return (
            "# Typedef: typedef struct\n\n"
            "### Задание №2\n\n"
            "- **Формулировка:**  \n"
            f"  Объявите структурный тип `{v.type_name}` так, чтобы переменную  \n"
            f"  можно было объявить как `{v.type_name} x;` без ключевого слова `struct`.  \n\n"
            "  Поля структуры (в указанном порядке):  \n"
            f"  {fields_str}  \n\n"
            "  Напишите **только одно объявление**.  \n"
        )

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="[скрыто]",
                expected="ok",
                compare_func=self._compare_default,
            )
        ]

    def run_solution(self, test: TestItem) -> Optional[tuple[str, str]]:
        ok, reason = _parse_typedef_struct(self.solution, self.variant)
        if ok:
            return None
        return reason, "[скрыто]"


import random
import os
from typing import Optional
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "base_module"))
from base_task import BaseTaskClass, TestItem


STRUCT_TYPES = ["int", "char", "float", "double", "short", "long"]


def _rng(seed: int) -> random.Random:

    return random.Random(seed)



# объявить структуру с заданными полями и вывести её sizeof.
class StructDeclarationTask(BaseTaskClass):



    TASK_TEMPLATE = """\

Объявите структуру `{name}` со следующими полями:
{fields_desc}

В функции `main` выведите `sizeof({name})`.
Пример вывода (значение зависит от платформы - главное что программа компилируется
и выводит одно целое число).
"""

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
    
        self._rng = _rng(seed)
        self._struct_name = ""
        self._fields: list[tuple[str, str]] = []  # список (тип, имя_поля)

    def generate_task(self) -> str:

        names = ["Point", "Person", "Record", "Item", "Node"]
        self._struct_name = self._rng.choice(names)

        n = self._rng.randint(2, 4)
        used_names: set[str] = set()
        self._fields = []
        field_name_pool = ["x", "y", "z", "age", "value", "count", "id", "key", "flag"]
        for _ in range(n):
            t = self._rng.choice(STRUCT_TYPES)
            # Исключаем уже использованные имена, чтобы не было дублей
            fn = self._rng.choice([f for f in field_name_pool if f not in used_names])
            used_names.add(fn)
            self._fields.append((t, fn))

        fields_desc = "\n".join(f"  - `{t} {fn}`" for t, fn in self._fields)
        return self.TASK_TEMPLATE.format(
            name=self._struct_name,
            fields_desc=fields_desc
        )

    def _generate_tests(self):
        
        def compare(output: str, _expected: str) -> bool:
            # Принимаем вывод, если это единственное положительное целое число
            output = output.strip()
            try:
                val = int(output)
                return val > 0
            except ValueError:
                return False

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected="Положительное целое (sizeof структуры)",
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
   
        return self._compile_internal()


# Проверяет инициализацию структуры. Нужно создать struct и правильно заполнить поля. Проверяется совпадение выведённых значений.

class StructInitTask(BaseTaskClass):


    TASK_TEMPLATE = """\


Объявите структуру `{name}`:
```c
struct {name} {{
{struct_fields}
}};
```

Создайте переменную типа `struct {name}` и инициализируйте её следующими значениями:
{init_values}

Выведите все поля через пробел в одну строку.
Ожидаемый вывод: `{expected}`
"""

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = _rng(seed)
        self._struct_name = "Data"
        self._fields: list[tuple[str, str, str]] = []  # (тип, имя, строковое_значение)
        self._expected = ""  

    def generate_task(self) -> str:

        self._struct_name = "Data"
        specs = [
            ("int",   "x",   str(self._rng.randint(-100, 100))),
            ("int",   "y",   str(self._rng.randint(-100, 100))),
            ("float", "val", f"{self._rng.uniform(-9.9, 9.9):.1f}"),
        ]
        self._fields = specs

        struct_fields = "\n".join(f"    {t} {n};" for t, n, _ in self._fields)
        init_values   = "\n".join(f"  - `{n}` = `{v}`" for _, n, v in self._fields)


        int_vals   = [v for t, _, v in self._fields if t == "int"]
        float_vals = [v for t, _, v in self._fields if t == "float"]
        self._expected = " ".join(int_vals + [f"{float(v):.1f}" for v in float_vals])

        return self.TASK_TEMPLATE.format(
            name=self._struct_name,
            struct_fields=struct_fields,
            init_values=init_values,
            expected=self._expected,
        )

    def _generate_tests(self):

        def compare(output: str, expected: str) -> bool:
            out_tokens = output.strip().split()
            exp_tokens = expected.strip().split()
            if len(out_tokens) != len(exp_tokens):
                return False
            for o, e in zip(out_tokens, exp_tokens):
                try:
                    if abs(float(o) - float(e)) > 1e-3:
                        return False
                except ValueError:
                    return False
            return True

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected=self._expected,
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()


#  прочитать из stdin значения, записать в поля структуры через '.' и '->', вывести сумму v.x + p->y.
class StructFieldAccessTask(BaseTaskClass):


    TASK_TEMPLATE = """\


Объявите структуру:
```c
struct Vec2 {{
    int x;
    int y;
}};
```

Программа должна:
1. Прочитать два целых числа `x` и `y` из stdin.
2. Создать переменную `struct Vec2 v` и присвоить поля через оператор `.`.
3. Создать указатель `struct Vec2 *p = &v`.
4. Вывести сумму `v.x + p->y`.

Формат ввода: два числа через пробел.
Формат вывода: одно число - сумма x и y.
"""

    def __init__(self, seed: int = 0, tests_num: int = 20, **kwargs):
        super().__init__(seed=seed, tests_num=tests_num, **kwargs)
        self._rng = _rng(seed)

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):

        self.tests = []
        for _ in range(self.tests_num):
            x = self._rng.randint(-500, 500)
            y = self._rng.randint(-500, 500)
            inp = f"{x} {y}"
            expected = str(x + y)

            # _x=x, _y=y фиксируют текущие значения итерации в замыкании
            def compare(output: str, exp: str, _x=x, _y=y) -> bool:
                return output.strip() == str(_x + _y)

            self.tests.append(TestItem(
                input_str=inp,
                showed_input=inp,
                expected=expected,
                compare_func=compare,
            ))

    def compile(self) -> Optional[str]:
        return self._compile_internal()



 # вывод alignof(char), alignof(int), alignof(double).


class StructAlignmentTask(BaseTaskClass):


    TASK_TEMPLATE = """\


Напишите программу, которая выводит требования к выравниванию (`alignof`)
следующих типов в следующем порядке, каждое на отдельной строке:
1. `alignof(char)`
2. `alignof(int)`
3. `alignof(double)`

Используйте заголовочный файл `<stdalign.h>` (C11) или `<stddef.h>`.
"""

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
        """
        Один тест без входных данных.
        compare() проверяет четыре условия:
          1. char: alignment == 1 (гарантировано стандартом C)
          2. int:  alignment in {2, 4} (зависит от ABI)
          3. double: alignment in {4, 8}
          4. монотонность: char <= int <= double
        """
        def compare(output: str, _expected: str) -> bool:
            tokens = output.strip().split()
            if len(tokens) != 3:
                return False
            try:
                char_a, int_a, dbl_a = int(tokens[0]), int(tokens[1]), int(tokens[2])
                return (
                    char_a == 1
                    and int_a in (2, 4)
                    and dbl_a in (4, 8)
                    and char_a <= int_a <= dbl_a  
                )
            except ValueError:
                return False

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected="1 4 8 (зависит от платформы)",
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()


# Написание структуры с меньшим sizeof за счёт правильного порядка полей
class StructPaddingTask(BaseTaskClass):

    TASK_TEMPLATE = """\

Дана «плохая» структура:
```c
struct Bad {{
    char  a;   // 1 байт + 3 байта padding (до int)
    int   b;   // 4 байта
    char  c;   // 1 байт + 7 байт padding (до double)
    double d;  // 8 байт
}};  // итого: 24 байта из-за padding
```

Напишите структуру `Good` с теми же полями (`char a`, `int b`, `char c`, `double d`),
но переупорядочив их так, чтобы `sizeof(Good)` было минимальным.

Программа должна вывести два числа через пробел:
1. `sizeof(struct Bad)`
2. `sizeof(struct Good)`

Ожидается, что второе число строго меньше первого.
"""

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
      
        # Принимаем любую пару (bad_sz, good_sz), где good_sz < bad_sz.
     
        def compare(output: str, _expected: str) -> bool:
            tokens = output.strip().split()
            if len(tokens) != 2:
                return False
            try:
                bad_sz, good_sz = int(tokens[0]), int(tokens[1])
                return good_sz < bad_sz and good_sz > 0
            except ValueError:
                return False

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected="24 16 (или аналогично: Good < Bad)",
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()


# Вывод размеров sizeof отдельных типов и структуры

class StructSizeofTask(BaseTaskClass):


    TASK_TEMPLATE = """\


Объявите структуру:
```c
struct Info {{
    int   id;
    float score;
    char  grade;
}};
```

Программа должна вывести через пробел:
1. `sizeof(int)`
2. `sizeof(float)`
3. `sizeof(char)`
4. `sizeof(struct Info)`

Пример вывода (x86-64): `4 4 1 12`
"""

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
 
        def compare(output: str, _expected: str) -> bool:
            tokens = output.strip().split()
            if len(tokens) != 4:
                return False
            try:
                si, sf, sc, ss = (int(t) for t in tokens)
                return (
                    si == 4
                    and sf == 4
                    and sc == 1
                    and ss >= si + sf + sc  
                )
            except ValueError:
                return False

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected="4 4 1 12",
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()

# Нужно записать float и вывести его байтовое представление(используя union).
class UnionUsageTask(BaseTaskClass):


    TASK_TEMPLATE = """\


Объявите union:
```c
union FloatBytes {{
    float f;
    unsigned char bytes[sizeof(float)];
}};
```

Программа читает одно вещественное число (float), записывает его в `u.f`,
затем выводит все `sizeof(float)` байт в шестнадцатеричном формате через пробел
(от байта 0 до байта sizeof(float)-1).

Формат ввода: одно число (float).
Формат вывода: `sizeof(float)` шестнадцатеричных значений через пробел.
Пример (для 1.0f на little-endian x86): `00 00 80 3f`
"""

    def __init__(self, seed: int = 0, tests_num: int = 10, **kwargs):
        super().__init__(seed=seed, tests_num=tests_num, **kwargs)
        self._rng = _rng(seed)

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
 
        import struct as _struct  

        self.tests = []
        test_floats = [1.0, -1.0, 0.5, 3.14, 0.0]
        for fval in test_floats[:self.tests_num]:
            raw = _struct.pack("<f", fval) 
            expected = " ".join(f"{b:02x}" for b in raw)

    
            def compare(output: str, exp: str, _exp=expected) -> bool:
                return output.strip().lower() == _exp.lower()

            self.tests.append(TestItem(
                input_str=str(fval),
                showed_input=str(fval),
                expected=expected,
                compare_func=compare,
            ))

    def compile(self) -> Optional[str]:
        return self._compile_internal()


# Записываем int, читаем его как массив байт.
class UnionMemoryOverlapTask(BaseTaskClass):


    TASK_TEMPLATE = """\


Объявите union:
```c
union Overlap {{
    int i;
    unsigned char b[sizeof(int)];
}};
```

Программа читает одно целое число (int), записывает его в `u.i`,
затем выводит все байты `u.b` через пробел в hex.

Демонстрирует, что запись в одно поле изменяет другое - поля хранятся
в одной и той же области памяти.

Формат ввода: одно целое число.
Формат вывода: 4 hex-байта через пробел (little-endian на x86).
"""

    def __init__(self, seed: int = 0, tests_num: int = 10, **kwargs):
        super().__init__(seed=seed, tests_num=tests_num, **kwargs)
        self._rng = _rng(seed)

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
    
        import struct as _struct

        self.tests = []
        values = [0, 1, 255, 256, -1, 0x12345678]
        for ival in values[:self.tests_num]:
            try:
                raw = _struct.pack("<i", ival) 
            except _struct.error:
                continue 
            expected = " ".join(f"{b:02x}" for b in raw)

            def compare(output: str, exp: str, _exp=expected) -> bool:
                return output.strip().lower() == _exp.lower()

            self.tests.append(TestItem(
                input_str=str(ival),
                showed_input=str(ival),
                expected=expected,
                compare_func=compare,
            ))

    def compile(self) -> Optional[str]:
        return self._compile_internal()



# Сравнить sizeof(struct) и sizeof(union) с одинаковыми полями
class UnionVsStructTask(BaseTaskClass):
    TASK_TEMPLATE = """\


Объявите:
```c
struct S {{
    int   a;
    float b;
    char  c;
}};

union U {{
    int   a;
    float b;
    char  c;
}};
```

Программа выводит через пробел:
1. `sizeof(struct S)`
2. `sizeof(union U)`

Ожидается, что `sizeof(union U) == sizeof(int)` (или `sizeof(float)`),
а `sizeof(struct S)` - сумма с padding.

Это демонстрирует главное отличие: union хранит только одно поле
одновременно, struct -все поля одновременно.
"""

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
        """
        Проверяем два инварианта:
          1. sizeof(union U) == 4  (max из {int=4, float=4, char=1})
          2. sizeof(struct S) > sizeof(union U)
        Если union != 4 - задание написано некорректно, возвращаем False.
        """
        def compare(output: str, _expected: str) -> bool:
            tokens = output.strip().split()
            if len(tokens) != 2:
                return False
            try:
                ss, su = int(tokens[0]), int(tokens[1])
                # union == max(sizeof полей) == 4; struct > union
                return ss > su and su == 4
            except ValueError:
                return False

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected="12 4 (или аналогично: struct > union)",
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()


# Вывести числовые значения констант enum
class EnumDefaultValuesTask(BaseTaskClass):


    TASK_TEMPLATE = """\

Объявите перечисление:
```c
enum Direction {{
    NORTH,
    EAST,
    SOUTH,
    WEST
}};
```

Программа должна вывести числовые значения каждой константы через пробел.
Ожидаемый вывод: `0 1 2 3`
"""

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):
        """
        Один тест с точным строковым сравнением.
        Значения enum по умолчанию не зависят от платформы (всегда 0, 1, 2, 3),
        поэтому допуск не нужен - используем точное сравнение строк.
        """
        expected = "0 1 2 3"

        def compare(output: str, exp: str) -> bool:
            return output.strip() == exp

        self.tests = [
            TestItem(
                input_str="",
                showed_input="(нет входных данных)",
                expected=expected,
                compare_func=compare,
            )
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()
    

#  Сопоставить числовой код и имя константы.
class EnumExplicitValuesTask(BaseTaskClass):
 

    TASK_TEMPLATE = """\

Объявите перечисление с явными значениями:
```c
enum HttpStatus {{
    OK       = 200,
    NOT_FOUND = 404,
    ERROR     = 500
}};
```

Программа читает одно целое число из stdin (200, 404 или 500)
и выводит соответствующее символическое имя:
- 200 → `OK`
- 404 → `NOT_FOUND`
- 500 → `ERROR`
- иначе → `UNKNOWN`

Формат ввода: одно целое число.
Формат вывода: строка с именем статуса.
"""

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = _rng(seed)

    def generate_task(self) -> str:
        return self.TASK_TEMPLATE

    def _generate_tests(self):

        cases = [
            ("200", "OK"),
            ("404", "NOT_FOUND"),
            ("500", "ERROR"),
            ("0",   "UNKNOWN"),   
            ("999", "UNKNOWN"),   
        ]

        def make_compare(exp: str):

            def compare(output: str, _exp: str) -> bool:
                return output.strip() == exp
            return compare

        self.tests = [
            TestItem(
                input_str=inp,
                showed_input=inp,
                expected=exp,
                compare_func=make_compare(exp),
            )
            for inp, exp in cases
        ]

    def compile(self) -> Optional[str]:
        return self._compile_internal()



TASK_REGISTRY: dict[str, type] = {

    "struct_declaration":    StructDeclarationTask,   # объявление struct
    "struct_init":           StructInitTask,           # инициализация struct
    "struct_field_access":   StructFieldAccessTask,    # доступ через . и ->
    "struct_alignment":      StructAlignmentTask,      # alignof (alignment)
    "struct_padding":        StructPaddingTask,        # padding + правильное хранение
    "struct_sizeof":         StructSizeofTask,         # sizeof(struct)

    "union_usage":           UnionUsageTask,           # использование union
    "union_memory_overlap":  UnionMemoryOverlapTask,   # перекрытие памяти
    "union_vs_struct":       UnionVsStructTask,        # отличие union от struct

    "enum_defaults":         EnumDefaultValuesTask,    # значения по умолчанию
    "enum_explicit":         EnumExplicitValuesTask,   # явное задание значений
}


if __name__ == "__main__":

  
    def run_example(task_key: str, task_cls, seed: int, student_code: str):
  
        print(f"Задача: {task_key}  [{task_cls.__name__}]  seed={seed}")
     

        task = task_cls(seed=seed)

    
        condition = task.init_task()
        print("Условие")
        print(condition)

     
        task.load_student_solution(solcode=student_code)

      
        ok, msg = task.check()


        print("Правильно" if ok else "Неправильно")
        print(msg)

  
    
  
    run_example(
        task_key="struct_declaration",
        task_cls=StructDeclarationTask,
        seed=0,
        student_code="""
#include <stdio.h>

struct Point {
    int x;
    char flag;
    double value;
};

int main(void) {
    printf("%zu\\n", sizeof(struct Point));
    return 0;
}
"""
    )

  


    run_example(
        task_key="struct_init",
        task_cls=StructInitTask,
        seed=0,
        student_code="""
#include <stdio.h>

struct Data {
    int x;
    int y;
    float val;
};

int main(void) {
    struct Data d = {42, -7, 3.5};
    printf("%d %d %.1f\\n", d.x, d.y, d.val);
    return 0;
}
"""
    )

  
 
    run_example(
        task_key="struct_field_access",
        task_cls=StructFieldAccessTask,
        seed=0,
        student_code="""
#include <stdio.h>

struct Vec2 {
    int x;
    int y;
};

int main(void) {
    struct Vec2 v;
    scanf("%d %d", &v.x, &v.y);
    struct Vec2 *p = &v;
    printf("%d\\n", v.x + p->y);
    return 0;
}
"""
    )

  

    run_example(
        task_key="struct_alignment",
        task_cls=StructAlignmentTask,
        seed=0,
        student_code="""
#include <stdio.h>
#include <stdalign.h>

int main(void) {
    printf("%zu\\n", alignof(char));
    printf("%zu\\n", alignof(int));
    printf("%zu\\n", alignof(double));
    return 0;
}
"""
    )

  

  
    run_example(
        task_key="struct_padding",
        task_cls=StructPaddingTask,
        seed=0,
        student_code="""
#include <stdio.h>

struct Bad {
    char   a;
    int    b;
    char   c;
    double d;
};

/* Оптимальный порядок: сначала большие типы, потом маленькие */
struct Good {
    double d;
    int    b;
    char   a;
    char   c;
};

int main(void) {
    printf("%zu %zu\\n", sizeof(struct Bad), sizeof(struct Good));
    return 0;
}
"""
    )

  

    run_example(
        task_key="struct_sizeof",
        task_cls=StructSizeofTask,
        seed=0,
        student_code="""
#include <stdio.h>

struct Info {
    int   id;
    float score;
    char  grade;
};

int main(void) {
    printf("%zu %zu %zu %zu\\n",
        sizeof(int),
        sizeof(float),
        sizeof(char),
        sizeof(struct Info));
    return 0;
}
"""
    )

  

    run_example(
        task_key="union_usage",
        task_cls=UnionUsageTask,
        seed=0,
        student_code="""
#include <stdio.h>

union FloatBytes {
    float f;
    unsigned char bytes[sizeof(float)];
};

int main(void) {
    union FloatBytes u;
    scanf("%f", &u.f);
    for (int i = 0; i < (int)sizeof(float); i++) {
        if (i > 0) printf(" ");
        printf("%02x", u.bytes[i]);
    }
    printf("\\n");
    return 0;
}
"""
    )

  

    run_example(
        task_key="union_memory_overlap",
        task_cls=UnionMemoryOverlapTask,
        seed=0,
        student_code="""
#include <stdio.h>

union Overlap {
    int i;
    unsigned char b[sizeof(int)];
};

int main(void) {
    union Overlap u;
    scanf("%d", &u.i);
    for (int j = 0; j < (int)sizeof(int); j++) {
        if (j > 0) printf(" ");
        printf("%02x", u.b[j]);
    }
    printf("\\n");
    return 0;
}
"""
    )

  
 
    run_example(
        task_key="union_vs_struct",
        task_cls=UnionVsStructTask,
        seed=0,
        student_code="""
#include <stdio.h>

struct S {
    int   a;
    float b;
    char  c;
};

union U {
    int   a;
    float b;
    char  c;
};

int main(void) {
    printf("%zu %zu\\n", sizeof(struct S), sizeof(union U));
    return 0;
}
"""
    )

  

    run_example(
        task_key="enum_defaults",
        task_cls=EnumDefaultValuesTask,
        seed=0,
        student_code="""
#include <stdio.h>

enum Direction {
    NORTH,
    EAST,
    SOUTH,
    WEST
};

int main(void) {
    printf("%d %d %d %d\\n", NORTH, EAST, SOUTH, WEST);
    return 0;
}
"""
    )


    run_example(
        task_key="enum_explicit",
        task_cls=EnumExplicitValuesTask,
        seed=0,
        student_code="""
#include <stdio.h>

enum HttpStatus {
    OK        = 200,
    NOT_FOUND = 404,
    ERROR     = 500
};

int main(void) {
    int code;
    scanf("%d", &code);
    switch (code) {
        case 200: printf("OK\\n");        break;
        case 404: printf("NOT_FOUND\\n"); break;
        case 500: printf("ERROR\\n");     break;
        default:  printf("UNKNOWN\\n");
    }
    return 0;
}
"""
    )

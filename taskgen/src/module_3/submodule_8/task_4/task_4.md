### Тема: Нарушение strict aliasing
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{func_name}({src_type} val)`, которая интерпретирует битовое представление числа `val` как тип `{dst_type}` и выводит его.
Запрещено использовать приведение типов через указатели: `*({dst_type}*)&val`.
Используйте безопасный способ: `memcpy` или `union`.

**Формат вывода:** `{fmt_str}`

**Уникальными значениями становятся:** `func_name`, `src_type`, `dst_type`, `fmt_str`
`seed % 2 == 0`: func_name = "float_to_int_bits", src_type = "float", dst_type = "int", fmt_str = "As int: %d\n"
`seed % 2 == 1`: func_name = "int_to_float_bits", src_type = "int", dst_type = "float", fmt_str = "As float: %g\n"

**Ввод:** Число передаётся в функцию как аргумент.
 
**Примеры решения**(для seed=15)
```c
// Вариант 1: memcpy 
#include <stdio.h>
#include <string.h>
void float_to_int_bits(float val) {
    int res;
    memcpy(&res, &val, sizeof(res));
    printf("As int: %d\n", res);
}

// Вариант 2: union
#include <stdio.h>
void float_to_int_bits(float val) {
    union { float f; int i; } u = { .f = val };
    printf("As int: %d\n", u.i);
}
```

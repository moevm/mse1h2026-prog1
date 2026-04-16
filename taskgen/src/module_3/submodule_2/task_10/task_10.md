### Тема: Указатель на указатель

**Сложность:** средняя

**Задание:**
Напишите функцию `void {func_name}(int **a, int **b)`, которая меняет местами два указателя через двойные указатели (обменивает адреса, а не значения). После обмена выведите значения, на которые теперь указывают `*a` и `*b`, соблюдая формат `{swap_output}`.

**Уникальными значениями становятся:** `func_name`, `swap_output`
`seed % 2 == 0`: func_name = "swap_double_ptrs", swap_output = "After swap: *a=`**a`, *b=`**b`"
`seed % 2 == 1`: func_name = "exchange_ptr_targets", swap_output = "Swapped => a:`**a` b:`**b`"
Ввод: Два двойных указателя передаются в функцию из `main()`.

**Пример:** (для seed=15 генерируются func_name = "exchange_ptr_targets", swap_output = "Swapped => a:`**a` b:`**b`")

**Пример решения:**
#include <stdio.h>

void exchange_ptr_targets(int **a, int **b) {
    int *temp = *a;
    *a = *b;
    *b = temp;
    printf("Swapped => a:%d b:%d\n", **a, **b);
}
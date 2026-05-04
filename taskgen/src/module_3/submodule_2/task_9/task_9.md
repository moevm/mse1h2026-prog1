# Тема: Сравнение указателей

**Сложность:** средняя

**Задание:**
Напишите функцию `void {func_name}(int *a, int *b)`, которая проверяет, указывают ли указатели на один и тот же адрес в памяти. Если адреса одинаковые — выведите сообщение `{msg_equal}`. Если адреса разные — сравните значения по указателям и выведите результат в формате `{format_cmp}`, подставив соответствующий оператор (`<`, `>` или `==`). 

**Уникальными значениями становятся:**
`func_name`, `msg_equal`, `format_cmp`
`seed % 2 == 0`: func_name = "compare_ptrs", msg_equal = "Same address", format_cmp = "`*a` <op> `*b"`
`seed % 2 == 1`: func_name = "ptr_equality_check", msg_equal = "Equal pointers", format_cmp = "Result: `*a` <op> `*b"`

**Ввод:** Указатели передаются в функцию из `main()`.

**Пример:** (для seed=15 генерируются `func_name` = "ptr_equality_check", `msg_equal` = "Equal pointers", `format_cmp` = ""Result: *a <op> *b"")

**Пример решения:**
```c
#include <stdio.h>

void ptr_equality_check(int *a, int *b) {
    if (a == b) {
        printf("Equal pointers\n");
    } else {
        const char *op;
        if (*a < *b) op = "<";
        else if (*a > *b) op = ">";
        else op = "==";
        printf("Result: %d %s %d\n", *a, op, *b);
    }
}
```
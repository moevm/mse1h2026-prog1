### Тема: Передача массива в функцию

**Сложность:** средня

**Задание:**
Напишите функцию `int {func_name}(int *arr, int size)`, которая принимает указатель на первый элемент массива и количество элементов. Используйте только арифметику указателей `*(arr + i)` для доступа к данным. Вычислите сумму всех элементов и выведите её в консоль, строго соблюдая формат `{print_format}`. 

**Уникальными значениями становятся:**
`func_name`, `print_format`
`seed % 2 == 0`: func_name = "calc_array_sum", print_format = "Sum: %d\n"
`seed % 2 == 1`: func_name = "ptr_sum_elements", print_format = "Total sum: %d\n"

**Ввод:** Массив и его размер передаются в функцию из `main()`.

**Пример:** (для seed=15 генерируются `func_name` = "ptr_sum_elements", `print_format` = "Total sum: %d\n")

**Пример решения:**
```c
#include <stdio.h>

void ptr_sum_elements(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += *(arr + i);
    }
    printf("Total sum: %d\n", sum);
}
```
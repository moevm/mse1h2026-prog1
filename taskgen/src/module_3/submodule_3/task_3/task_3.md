Тема: Сравнение размеров (указатель vs массив)
Сложность: средняя
Задание:
Напишите функцию `void {func_name}(int size)`, которая объявляет массив `{type} arr[size]` и указатель на его начало. Выведите размер массива в байтах, размер указателя и размер одного элемента массива, строго соблюдая заданный формат. Запрещено использовать глобальные переменные и библиотеки кроме `<stdio.h>`. Для вычисления размеров используйте оператор `sizeof`.

Формат вывода:
Array size: X
Pointer size: Y
Element size: Z

**Уникальными значениями становятся:**
`func_name`, `type`, `size`
`seed % 3 == 0`: func_name = "show_sizes_int", type = "int", size = seed % 30
`seed % 3 == 1`: func_name = "show_sizes_float", type = "float", size = seed % 30
`seed % 3 == 2`: func_name = "show_sizes_char", type = "char", size = seed % 30

**Пример:** (для seed=15 генерируются `func_name` = "show_sizes_float", `type` = "float", `size` = 15)

**Пример решения:**f
```c
#include <stdio.h>

void show_sizes_float(int size) {
    float arr[15];
    float *ptr = arr;
    
    printf("Array size: %zu\n", sizeof(arr));
    printf("Pointer size: %zu\n", sizeof(ptr));
    printf("Element size: %zu\n", sizeof(arr[0]));
}
```
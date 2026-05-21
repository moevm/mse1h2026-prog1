### Тема: Использование неинициализированной памяти
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{func}(int size)`, которая выделяет память для массива из `size` элементов типа `{elem}` и гарантирует, что все элементы равны `{init_val}` до начала использования. Верните указатель на выделенную память. Используйте только стандартную библиотеку C.

**Формат вывода:** функция возвращает указатель, инициализация проверяется автотестом.

**Уникальными значениями становятся:** `func_name`, `elem_type`, `init_value`
`seed % 3 == 0`: func_name = "create_zero_array", elem_type = "int", init_value = "0"
`seed % 3 == 1`: func_name = "init_float_array", elem_type = "float", init_value = "0.0f"
`seed % 3 == 2`: func_name = "alloc_cleared_buf", elem_type = "char", init_value = "0"

**Ввод:** Размер массива `size` передаётся в функцию как аргумент.

**Пример решения:**(для seed=15)
```c
#include <stdlib.h>

int* create_zero_array(int size) {
    int *arr = calloc(size, sizeof(int));
    return arr;
}
```
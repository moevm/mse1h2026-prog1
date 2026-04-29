Тема: "Функция realloc()"
Сложность: средняя
Задание:
Создайте функцию `func_name(int initial_size, int new_size)`, которая выделяет память для массива из `initial_size` элементов типа `type` с помощью `initial_func`. Инициализируйте существующие элементы значениями по формуле: `arr[i] = init_formula`. Измените размер массива на `new_size` элементов с помощью `realloc`. Заполните *только новые* элементы значениями по формуле: `arr[i] = new_formula`. Выведите все элементы итогового массива в формате `Array: val1, val2, ..., valN`. Освободите память с помощью `free()`. Проверка результатов выделения и изменения памяти на `NULL` обязательна: при ошибке `realloc` освободите исходный блок, выведите `Allocation failed` и завершите работу функции. 
Формат вывода:
`Array: val1, val2, ..., valN`

**Уникальными значениями становятся:**
`func_name`, `initial_func`, `type`, `init_formula`, `new_formula`
`seed % 2 == 0`: func_name = "realloc_int", initial_func = "malloc", type = int, init_formula = "(seed + i) + 1", new_formula = "(seed + i) * 2"
`seed % 2 == 1`: func_name = "realloc_float", initial_func = "calloc", type = float, init_formula = "(seed + i) * 1.5", new_formula = "(seed + i) + 10.0"

**Ввод:** Размеры `initial_size` и `new_size` передаются в функцию как аргументы из `main()`. 

**Пример:** (для seed 15 генерируются `func_name` = "realloc_float", `initial_func` = "calloc", `type` = float, `init_formula` = "(seed + i) * 1.5", `new_formula` = "(seed + i) + 10.0")

**Пример решения:**
```c
#include <stdio.h>
#include <stdlib.h>

void realloc_float(int initial_size, int new_size) {
    if (initial_size <= 0 || new_size <= 0) return;
    
    float *arr = calloc(initial_size, sizeof(float));
    if (arr == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return;
    }

    for (int i = 0; i < initial_size; i++) {
        arr[i] = i * 1.5f;
    }

    float *tmp = realloc(arr, new_size * sizeof(float));
    if (tmp == NULL) {
        free(arr);
        fprintf(stderr, "Allocation failed\n");
        return;
    }
    arr = tmp;

    for (int i = initial_size; i < new_size; i++) {
        arr[i] = i + 10.0f;
    }

    printf("Array: ");
    for (int i = 0; i < new_size; i++) {
        printf("%g", arr[i]);
        if (i < new_size - 1) {
            printf(", ");
        }
    }
    printf("\n");

    free(arr);
}
```
### Тема: "Функция calloc()"
**Сложность:** средняя
**Задание:**
Создайте функцию `func_name(int size)`, которая выделяет память с помощью `calloc` для массива из `size` элементов типа `type`. Инициализируйте массив значениями по формуле: `arr[i] = {formula}`. Выведите все элементы массива в формате `Array: val1, val2, ..., valN`. Освободите память с помощью `free()`. Проверка результата `calloc` на `NULL` обязательна: если выделение памяти оказалось неудачным, выведите `Allocation failed` в `stderr` и завершите работу функции.
Формат вывода:
`Array: val1, val2, ..., valN`


**Уникальными значениями становятся:**
`func_name`, `type`, `formula`
`func_name`, `type`, `formula`
`seed % 2 == 0`: func_name = "init_calloc_array", type = int, formula = "(seed + i) + 1"
`seed % 2 == 1`: func_name = "process_calloc_buf", type = float, formula = "(seed + i) * 1.5"

**Ввод:** Размер массива `size` передаётся в функцию как аргумент из `main()`. 

**Пример:** (для seed 15 генерируются `func_name` = "process_calloc_array", `type` = float, `formula` = "(seed + i) * 1.5")

**Пример решения:**
```c
#include <stdio.h>
#include <stdlib.h>

void process_calloc_array(int size) {
    if (size <= 0) return;
    
    float *arr = calloc(size, sizeof(float));
    if (arr == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        arr[i] = i * 1.5f;
    }

    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%g", arr[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("\n");

    free(arr);
}
```
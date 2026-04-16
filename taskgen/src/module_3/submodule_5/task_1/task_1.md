### Тема: "Функция malloc()"
**Сложность:** средняя

**Задание:**
 Реализуйте функцию `alloc_and_print(int size)`, которая внутри себя выделяет память с помощью `malloc` для массива из `size` элементов типа `type`. Инициализируйте массив значениями по формуле: `arr[i] = value * i`. Выведите все элементы массива в формате `Array: val1, val2, ..., valN`. Освободите выделенную память с помощью `free()`. Проверка результата `malloc` на `NULL` обязательна: если выделение памяти оказалось неудачным, выведите `Allocation failed` в `stderr` и завершите работу функции. 
 Формат вывода:
 `Array: v1, v2, ..., vN`

**Уникальными значениями становятся:**
`type`, `value`
`seed % 2 == 0`: type = int, value = seed
`seed % 2 == 1`: type = float, value = seed / 2

**Ввод:** 
Размер массива `size` передаётся в функцию как аргумент из `main()`. 

**Пример:** (для seed 15 генерируются `type` = float, `value` = 7.5)

**Пример решения:**
```c
#include <stdio.h>
#include <stdlib.h>

void alloc_and_print(int size) {
    if (size <= 0) return;
    
    float *arr = malloc(size * sizeof(float));
    if (arr == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        arr[i] = 7.5f * i;
    }

    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%g", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("\n");

    free(arr);
}
```
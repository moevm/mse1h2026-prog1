# Тема: "Функция calloc()"
**Сложность:** средняя

## Задание:
Выделите память с помощью функции `calloc` для массива из `{count}` элементов типа `{type}`.
Инициализируйте массив значениями по формуле: `arr[i] = {formula}`.
Выведите все элементы массива.
Освободите память с помощью `free()`.

**Формат вывода:**
`Array: val1, val2, ..., valN`

## Уникальные значения (по seed):
seed % 2 == 0: type = int, count = 5, formula = "i + 1" 
seed % 2 == 1: type = float, count = 10, formula = "i * 1.5" 

**Пример решения:**
```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int *arr = (int*)calloc(5, sizeof(int));
    int i;

    for (i = 0; i < 5; i++) {
        arr[i] = i + 1;
    }

    printf("Array: ");
    for (i = 0; i < 5; i++) {
        printf("%d", arr[i]);
        if (i < 4) {
            printf(", "); 
        }
    }
    printf("\n");
    
    free(arr);
    return 0;
}
```
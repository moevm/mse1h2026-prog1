# Тема: Разыменовывание

## Варианты задач (определяются по seed)

### seed % 2 == 0 : Задача 2.4.1
 - Задание: Напишите функцию void sum_pointers(int *a, int *b, int *result), которая записывает сумму значений по адресу a и b соответственно по адресу result.
 - Формат вывода:
   *result
 - Пример решения:
```c
#include <stdio.h>

void sum_pointers(int *a, int *b, int *result) {
    *result = *a + *b;
    printf('%d\n', *result);
}
```

### seed % 2 == 1: 
 - Задание: Напишите функцию void make_absolute(int *ptr), которая возвращает значение *ptr по модулю.
 - Формат вывода:
   *ptr
 - Пример решения:
```c
#include <stdio.h>

void make_absolute(int *ptr) {
    if (*ptr < 0) {
        *ptr = -*ptr;
    }
    printf('%d\n', *ptr);
}
```
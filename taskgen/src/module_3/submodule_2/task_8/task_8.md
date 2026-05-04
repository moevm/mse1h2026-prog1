# Тема: "Арифметика указателей"

**Сложность:** средняя

**Задание:**
Напишите функцию, на вход которой подается указатель на массив arr состоящий из 10 элементов типа {type}.
Требуется увеличить каждый элемент, находящийся на {count} позиции, на {add_value}.
Используйте только арифметику указателей для доступа к элементам.
Формат вывода:
Array after modification: val1, val2, ..., val10
 
**Уникальными значениями становятся:** 
`type`, `count`, `add_value`.

seed % 2 == 0: type = int; count  = "каждой позиции кратной seed % 10"; add_value = seed % 1000
seed % 2 == 1: type = float; count = "каждой позиции не кратной seed % 10"; add_value = (seed % 1000) * 0.5

**Ввод:** массив задаётся в коде.

**Пример для seed=15:**
  - type = float (15 % 2 = 1)
  - count = "на каждой позиции кратной 5" 
  - add_value = 7.5

**Пример решения:**
```c
#include <stdio.h>

void func_name(float *arr) {
    int step = 5;
    float add_value = 7.5f;
    
    for (int i = 0; i < 10; i++) {
        if (i % step == 0) {
            *(arr + i) += add_value; 
        }
    }
    
    printf("Array after modification: ");
    for (int i = 0; i < 10; i++) {
        printf("%f", *(arr + i));
        if (i < 9) {
            printf(", ");
        }
    }
    printf("\n");
}
``` 
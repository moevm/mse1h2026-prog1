## Тема: Сравнение размеров (указатель vs массив)
**Сложность:** средняя

## Задание:
Объявите массив `int arr[10]` и указатель на начало массива.
Выведите:
 - размер массива
 - размер указателя
 - размер одного элемента

**Формат вывода:**
Array size: X
Pointer size: Y
Element size: Z

**Пример решения:**
```c
#include <stdio.h>

int main() {
    int arr[10];
    int *ptr = arr;
    
    printf("Array size: %zu\n", sizeof(arr));
    printf("Pointer size: %zu\n", sizeof(ptr));
    printf("Element size: %zu\n", sizeof(arr[0]));
    
    return 0;
}
```
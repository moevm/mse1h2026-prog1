### Тема: Разница между массивом и указателем

**Сложность:** средняя

**Задание:**
Напишите функцию `void {func_name}(int *arr, int size)`, которая обходит массив с помощью указателя и выводит каждый элемент. Имя массива является неизменяемым адресом (`arr++` недопустимо), поэтому для обхода создайте вспомогательный указатель `int *p = arr` или используйте арифметику `*(arr + i)`. Запрещено использовать синтаксис `arr[i]`. 
Формат вывода:
{print_format}

**Уникальными значениями становятся:** `func_name`, `print_format`
`seed % 2 == 0`: func_name = "traverse_with_ptr", print_format = "Element: %d\n"
`seed % 2 == 1`: func_name = "print_via_pointer", print_format = "Val: %d\n"

**Пример:** (для seed=15 генерируются `func_name` = "print_via_pointer", `print_format` = "Val: %d\n")

**Пример решения:**
```c
#include <stdio.h>

void print_via_pointer(int *arr, int size) {
    int *p = arr;
    for (int i = 0; i < size; i++) {
        printf("Val: %d\n", *p);
        p++;
    }
}
```

# Тема: "Функция malloc()"
**Сложность:** средняя
**Задание:** 
Выделите память с помощью функции malloc на переменную типа type, инициализируйте переменную значением value, выведите значение переменной. Освободите память с помощью `free()`.

**Формат вывода:**
Value: X

**Уникальными значениями становятся: type, value:**
 seed % 2 == 0: type = int, value = seed
 seed % 2 == 1: type = float, value = seed / 2


**Пример решения**
```c
#include <stdio.h>
#include <stdlib.h>   

int main(void)
{    int *ptr = (int*)malloc(sizeof(int));
     *ptr = 24;     
     printf("Value: %d\n", *ptr);
     free(ptr);
}
```

# Тема: Адрес переменной 

### seed % 2 == 0 : 
**Условие:**
Напишите функцию `void solution(int *ptr, int new_value)`, которая:
1. Выведет значение *ptr до изменения
2. Изменит значение по адресу ptr на new_value
3. Выведет значение *ptr после изменения

**Формат вывода:**
Before: X
After: Y

**Пример решения:**
```c
#include <stdio.h>

void solution(int *ptr, int new_value) {
    printf("Before: %d\n", *ptr);
    *ptr = new_value;
    printf("After: %d\n", *ptr);
}
```

### seed % 2 == 1: 

**Условие:**
Чтение переменной через указатель

Напишите функцию void print_value(int *ptr), которая принимает указатель на целое число 
и выводит значение, находящееся по этому адресу. 
Если указатель NULL вывести Value: 0

**Формат вывода:**
Value: X

**Пример решения:**
```c
#include <stdio.h>

void print_value(int *ptr) {
    if (ptr == NULL){
        printf("Value: 0\n");
    }
    else{
    printf("Value: %d\n", *ptr);
    }
}
```




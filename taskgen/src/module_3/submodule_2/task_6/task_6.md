### Тема: Адрес переменной 

**Сложность:** легкая

**Задача:**
Напишите функцию `void {func_name}({type} *ptr, {type} new_value)`, которая:
Выведет значение *ptr до изменения. Изменит значение по адресу ptr на new_value. Выведет значение *ptr после изменения
Формат вывода:
{print_format}

**Уникальными значениями становятся:**
func_name, type, print_format

seed % 2 == 0: func_name = change_value; type = float; print_format = Value changed from `old_val` to `new_val`:

seed % 2 == 1: func_name = transform_number; type = int; print_format = 
Before: `old_value`
After: `new_value`

**Пример решения:**
```c
#include <stdio.h>

void transform_number(int *ptr, int new_value) {
    printf("Before: %d\n", *ptr);
    *ptr = new_value;
    printf("After: %d\n", *ptr);
}
```

 





# Тема: Адрес переменной 

**Задача:**
Чтение переменной через указатель
Напишите функцию void {func_name}(int *ptr), которая принимает указатель на целое число 
и выводит значение, находящееся по этому адресу. 
Если указатель NULL вывести {null_msg}.
Формат вывода:
{print_format}

**Уникальные значения:**
func_name, print_format, null_msg
seed % 4 == 0: func_name = "read_ptr_value", print_format = "Value: %d\n", null_msg = "Value: 0\n"
seed % 4 == 1: func_name = "print_deref", print_format = "Dereferenced: %d\n", null_msg = "NULL pointer\n"
seed % 4 == 2: func_name = "show_content", print_format = "Content: %d\n", null_msg = "No data\n"
seed % 4 == 3: func_name = "fetch_from_ptr", print_format = "Ptr value: %d\n", null_msg = "Empty\n"

**Пример решения:**
```c
#include <stdio.h>

void fetch_from_ptr(int *ptr) {
    if (ptr == NULL) {
        printf("Empty\n");
        return;
    }
    printf("Ptr value: %d\n", *ptr);
}
```




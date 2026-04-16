### Тема: Разыменовывание

# Сложность: легкая

# Задание:
 Напишите функцию void {func_name}(int *{param_name}), которая изменяет значение по адресу {param_name} на его модуль (абсолютное значение). После изменения выведите полученное значение в формате {fmt_str}.

  - Уникальными значениями становятся:
  func_name, param_name, fmt_str
  
  - Правила генерации параметров (по seed):
 seed % 3 == 0: func_name = "make_absolute", param_name = "ptr"
 seed % 3 == 1: func_name = "normalize_value", param_name = "val_ptr"
 seed % 3 == 2: func_name = "get_abs_ref", param_name = "num"
 seed % 4 == 0: fmt_str = "Result: %d\n"
 seed % 4 == 1: fmt_str = "Модуль => %d\n"
 seed % 4 == 2: fmt_str = "[ABS] %d\n"
 seed % 4 == 3: fmt_str = "Absolute value: %d\n"

  - Ввод: В функцию передаётся адрес целочисленной переменной (значения инициализируются в main, *ptr = +-seed % 100).


# Пример: 
 для seed=15 генерируются: func_name = "make_absolute", param_name = "ptr", fmt_str = "Absolute value: %d\n" (15 % 3 = 0, 15 % 4 = 3)

 - Ввод:
 *ptr = -15
 
 - Вывод:
 Absolute value: 12

 - Пример решения:
```c
#include <stdio.h>

void make_absolute(int *ptr) {
    if (*ptr < 0) {
        *ptr = -*ptr;
    }
    printf("Absolute value: %d\n", *ptr);
}

int main(void) {
    int x = -12;
    make_absolute(&x);
    return 0;
}
```
  


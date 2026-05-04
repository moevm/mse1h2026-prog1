### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `int {func_name}(char *s)`, которая возвращает количество символов в строке (без учёта `'\0'`).

**Уникальными значениями становятся:**
func_name
seed % 4 == 0: func_name = my_strlen
seed % 4 == 1: func_name = calc_str_len
seed % 4 == 2: func_name = get_string_size
seed % 4 == 3: func_name = ptr_strlen


**Пример решения:**
```c
#include <stdio.h>

int my_strlen(char *s) {
    int len = 0;
    while (*s != '\0') {
        len++;
        s++;
    }
    return len;
}
```


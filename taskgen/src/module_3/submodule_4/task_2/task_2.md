### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *dest, char *src)`, которая копирует строку `src` в `dest` включая `'\0'`.

**Уникальными значениями становятся:**
func_name
seed % 4 == 0: func_name = "my_strcpy"
seed % 4 == 1: func_name = "custom_strcpy"
seed % 4 == 2: func_name = "str_copy"
seed % 4 == 3: func_name = "manual_strcpy"

**Пример решения:**
```c
#include <stdio.h>

char *my_strcpy(char *dest, char *src) {
    char *orig_dest = dest;
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
    return orig_dest;
}
```


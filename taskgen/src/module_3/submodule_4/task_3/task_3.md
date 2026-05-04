### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *dest, char *src)`, которая приписывает строку `src` к концу `dest`.

**Уникальными значениями становятся:**
func_name
seed % 4 == 0: func_name = "my_strcat"
seed % 4 == 1: func_name = "custom_strcat"
seed % 4 == 2: func_name = "append_string"
seed % 4 == 3: func_name = "manual_strcat"

**Пример решения:**
```c
#include <stdio.h>

char *my_strcat(char *dest, char *src) {
    char *orig_dest = dest;
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
    return orig_dest;
}
```



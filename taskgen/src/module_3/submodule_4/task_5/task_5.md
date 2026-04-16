### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *s, char c)`, которая находит первое вхождение символа `c` в строке `s`.

**Уникальными значениями становятся:**
func_name
seed % 4 == 0: func_name = "my_strchr"
seed % 4 == 1: func_name = "custom_strchr"
seed % 4 == 2: func_name = "find_first_char"
seed % 4 == 3: func_name = "manual_strchr"

**Пример решения:**
```c
#include <stdio.h>

char *my_strchr(char *s, char c) {
    while (*s != '\0') {
        if (*s == c) {
            return s;
        }
        s++;
    }
    return NULL;
}
```





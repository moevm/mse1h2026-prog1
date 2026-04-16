### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `int {func_name}(char *s1, char *s2)`, которая сравнивает две строки.

**Уникальными значениями становятся:**
func_name
seed % 4 == 0: func_name = "my_strcmp"
seed % 4 == 1: func_name = "custom_strcmp"
seed % 4 == 2: func_name = "str_compare"
seed % 4 == 3: func_name = "manual_strcmp"

**Пример решения:**
```c
#include <stdio.h>

int my_strcmp(char *s1, char *s2) {
    while (*s1 != '\0' && *s2 != '\0') {
        if (*s1 != *s2) {
            return *s1 - *s2;
        }
        s1++;
        s2++;
    }
    return *s1 - *s2;
}
```





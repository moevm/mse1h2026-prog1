### Тема: Нультерминаторные строки
**Сложность:** средняя

**Задание:**
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).
Напишите функцию `char *{func_name}(char *haystack, char *needle)`, которая находит первое вхождение подстроки `needle` в строке `haystack`.

**Уникальными значениями становятся:**
func_name
seed % 4 == 0: func_name = "my_strstr"
seed % 4 == 1: func_name = "custom_strstr"
seed % 4 == 2: func_name = "find_substring"
seed % 4 == 3: func_name = "manual_strstr" 

**Пример решения:**
```c
#include <stdio.h>

char *my_strstr(char *haystack, char *needle) {
    if (*needle == '\0') return haystack;
    
    while (*haystack != '\0') {
        char *h = haystack;
        char *n = needle;
        while (*h != '\0' && *n != '\0' && *h == *n) {
            h++;
            n++;
        }
        if (*n == '\0') return haystack;
        haystack++;
    }
    return NULL;
}
```



# Тема: "Нультерминаторные строки*"
**Сложность:** средняя

## Задание:
Реализуйте функцию из библиотеки `string.h` самостоятельно (без использования библиотечной версии).

## Варианты задач (определяются по seed)

### seed % 6 == 0 : Задача 2.14.1
**Условие:**
Напишите функцию `int my_strlen(char *s)`, которая возвращает количество символов в строке (без учёта `'\0'`).

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

### seed % 6 == 1 :
**Условие:**
Напишите функцию `char *my_strcpy(char *dest, char *src)`, которая копирует строку `src` в `dest` включая `'\0'`.

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

### seed % 6 == 2 : 
**Условие:**
Напишите функцию `char *my_strcat(char *dest, char *src)`, которая приписывает строку `src` к концу `dest`.

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


### seed % 6 == 3 :
**Условие:**
Напишите функцию `int my_strcmp(char *s1, char *s2)`, которая сравнивает две строки.

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

### seed % 6 == 4 : 
**Условие:**
Напишите функцию `char *my_strchr(char *s, char c)`, которая находит первое вхождение символа `c` в строке `s`.


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

### seed % 6 == 5 :
**Условие:**
Напишите функцию `char *my_strstr(char *haystack, char *needle)`, которая находит первое вхождение подстроки `needle` в строке `haystack`.

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



### Тема: Опасность переполнения буфера

# Сложность: средняя

# Задание:
 Реализуйте функцию {func_name}(char *dest, size_t dest_size, const char *src). Функция должна безопасно скопировать строку src в буфер dest, гарантируя, что будет записано не более dest_size - 1 символов, а в конце буфера всегда будет стоять завершающий '\0'. Запрещено использовать стандартные строковые функции (strcpy, strncpy, memcpy, snprintf и т.д.). Реализация должна корректно обрабатывать случай dest_size == 0 или dest_size == 1. Внутри функции необходимо определить статус операции: {label_ok} - строка поместилась целиком, {label_cut} - строка была обрезана, и вывести результат соответствующий формату вывода.
 Формат вывода:
 {print_format}

 - Уникальными значениями становятся:
 `func_name` (имя функции)
 ["bounded_strcpy", "str_copy_safe"] 
 `print_format` (строка формата вывода)
 ["Status: 5 copied | Data: Hello | Flag: OK"; "Result: len=5 content=Hello check=FULL"]
 `label_ok` (статус при успешном копировании)
 ["OK", "FULL"]
 `label_cut` (статус при обрезке строки)
 ["TRUNC", "CUT"]


 - Ввод: значения передаются в функцию через аргументы `dest`, `dest_size` и `src`. В `main()` параметры задаются явно.

 - Пример: (для seed 15 генерируются `func_name` = "str_copy_safe", `print_format` = "Result: len=5 content=Hello check=FULL", 
 `label_ok` = "FULL", `label_cut` = "CUT")

Пример решения:
```c
#include <stddef.h>
#include <stdio.h>

size_t str_copy_safe(char *dest, size_t dest_size, const char *src) {
    if (dest == NULL || src == NULL || dest_size == 0) return 0;

    size_t i = 0;
    while (i < dest_size - 1 && src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';

    const char *ok_lbl = "FULL";
    const char *cut_lbl = "CUT";
    const char *status = (src[i] == '\0') ? ok_lbl : cut_lbl;

    printf("Result: len=%zu content='%s' check=%s\n", i, dest, status);
    return i;
}
```
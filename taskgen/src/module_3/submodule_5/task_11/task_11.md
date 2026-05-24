### Тема: Защита от Double Free
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{func_name}({type} **{ptr_name})`, которая безопасно освобождает память и предотвращает двойное освобождение.
Функция должна:
1. Проверить, что `{ptr_name}` не равен `NULL` и не указывает на `NULL`.
2. Освободить память через `free()`.
3. Присвоить `{ptr_name}` значение `NULL`, чтобы указатель в вызывающем коде стал безопасным.

**Уникальными значениями становятся:** `func_name`, `type`, `ptr_name`
`seed % 3 == 0`: func_name = "safe_free_int", type = "int", ptr_name = "buffer"
`seed % 3 == 1`: func_name = "safe_free_float", type = "float", ptr_name = "data"
`seed % 3 == 2`: func_name = "safe_free_char", type = "char", ptr_name = "str"

**Ввод:** Адрес указателя передаётся в функцию. Драйвер создаёт массив, вызывает функцию, проверяет обнуление и вызывает повторно.

**Пример решения:** (для seed=15)
```c
#include <stdlib.h>

void safe_free_int(int **buffer) {
    if (buffer && *buffer) {
        free(*buffer);
        *buffer = NULL;
    }
}
```


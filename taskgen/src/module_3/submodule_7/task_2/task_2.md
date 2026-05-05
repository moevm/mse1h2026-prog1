### Тема: const correctness
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{func_name}({src_type} const *{src_param}, {dst_type} *{dst_param}, int size)`, которая копирует элементы из массива `{src_param}` в `{dst_param}` и выводит пары "оригинал → копия" в формате `{fmt_str}`.
- Исходный массив `{src_param}` должен быть доступен **только для чтения** (`const` обязателен).
- Массив `{dst_param}` должен оставаться **изменяемым** (без `const`).
- Нарушение контракта приведёт к ошибке компиляции в автотесте.

**Формат вывода:** `{fmt_str}` (для каждого индекса)

**Уникальными значениями становятся:** `func_name`, `src_type`, `dst_type`, `src_param`, `dst_param`, `fmt_str`
`seed % 3 == 0`: func_name = "copy_ints", src_type = "int", dst_type = "int", src_param = "src", dst_param = "dst", fmt_str = "Copy[%d]: %d -> %d\n"
`seed % 3 == 1`: func_name = "scale_floats", src_type = "float", dst_type = "float", src_param = "input", dst_param = "output", fmt_str = "Scaled[%d]: %.1f -> %.1f\n"
`seed % 3 == 2`: func_name = "mirror_chars", src_type = "char", dst_type = "char", src_param = "read_buf", dst_param = "write_buf", fmt_str = "Mir[%d]: '%c' -> '%c'\n"

**Ввод:** Массивы и размер передаются в функцию как аргументы. Драйвер инициализирует `{src_param}` и вызывает функцию.

**Пример решения:** (для seed=15)
```c
#include <stdio.h>

void copy_ints(const int *src, int *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
        printf("Copy[%d]: %d -> %d\n", i, src[i], dst[i]);
    }
}
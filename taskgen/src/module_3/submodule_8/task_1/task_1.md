### Тема: Разыменование `NULL`
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{p["func_name"]}({p["elem_type"]} *ptr)`, которая безопасно разыменовывает указатель. Если `ptr == NULL`, функция должна вывести `{p["null_msg"]}` и вернуть `{p["default_val"]}`. В противном случае — вывести `{p["ok_msg"]}` и вернуть значение `*ptr`.

**Формат вывода:**
- При `ptr == NULL`: `{fmt_null}`
- При `ptr != NULL`: `{fmt_ok}`

**Уникальными значениями становятся:** `func_name`, `elem_type`, `ok_msg`, `null_msg`, `elem_format`, `default_val`
`seed % 3 == 0`: func_name = "safe_deref_int", elem_type = "int", ok_msg = "Value: ", null_msg = "NULL ptr: ", format = "%d\\n", default = "0"
`seed % 3 == 1`: func_name = "read_float_safe", elem_type = "float", ok_msg = "Got: ", null_msg = "Error: ", format = "%.1f\\n", default = "0.0f"
`seed % 3 == 2`: func_name = "get_char_checked", elem_type = "char", ok_msg = "Char: ", null_msg = "No data: ", format = "%c\\n", default = "'\\0'"

**Ввод:** Указатель передаётся в функцию как аргумент. 

**Пример решения:** (для seed=15)
```c
#include <stdio.h>

int safe_deref_int(int *ptr) {
    if (ptr == NULL) {
        printf("NULL ptr: %d\n", 0);
        return 0;
    }
    printf("Value: %d\n", *ptr);
    return *ptr;
}
```
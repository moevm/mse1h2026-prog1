### Тема: Висячий указатель
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{func_name}({type} {val_name})`, которая возвращает указатель на переданное значение `{val_name}`. Запрещено возвращать адрес локальной переменной (`return &{val_name};`) — это создаст висячий указатель. Обязательно используйте динамическое выделение памяти (`malloc`/`calloc`), скопируйте значение и верните указатель. При ошибке выделения верните `NULL`.

**Формат вывода:** функция возвращает указатель, проверка выполняется драйвером.

**Уникальными значениями становятся:** `func_name`, `type`, `val_name`, `fmt`
`seed % 3 == 0`: func_name = "safe_int_ptr", type = "int", val_name = "val", fmt = "%d\n"
`seed % 3 == 1`: func_name = "safe_float_ptr", type = "float", val_name = "x", fmt = "%.2f\n"
`seed % 3 == 2`: func_name = "safe_char_ptr", type = "char", val_name = "c", fmt = "%c\n"

**Ввод:** Значение передаётся в функцию как аргумент. 

**Пример решения:** (для seed=15)
```c
#include <stdlib.h>

int* safe_int_ptr(int val) {
    int *p = malloc(sizeof(int));
    if (p) *p = val;
    return p;
}
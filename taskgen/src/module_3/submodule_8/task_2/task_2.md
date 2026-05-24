### Тема: Выход за границы
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {func_name}({elem_type} *arr, int size, int index)`, которая безопасно обращается к элементу массива.
- Если `index` выходит за допустимые пределы `[0, size)`, функция должна вывести `{oom_msg}`.
- В противном случае — вывести `{ok_msg}` и значение `arr[index]`.

**Формат вывода:**
- При успехе: `{ok_msg}{elem_format}`
- При ошибке: `{oom_msg}{elem_format}`

**Уникальными значениями становятся:** `func_name`, `elem_type`, `ok_msg`, `oom_msg`, `elem_format`
`seed % 2 == 0`: func_name = "safe_get_int", elem_type = "int", ok_msg = "OK: ", oom_msg = "OOB: ", elem_format = "%d\\n"
`seed % 2 == 1`: func_name = "safe_get_float", elem_type = "float", ok_msg = "Value: ", oom_msg = "Error: ", elem_format = "%.2f\\n"

**Ввод:** Массив, размер и индекс передаются в функцию как аргументы.

**Пример решения:** (для seed=15)
```c
#include <stdio.h>

void safe_get_float(float *arr, int size, int index) {
    if (index < 0 || index >= size) {
        printf("Error: %.2f\n", 0.0f);
    } else {
        printf("Value: %.2f\n", arr[index]);
    }
}
```
### Тема: Выход за границы
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{p["func_name"]}({p["elem_type"]} *arr, int size, int index)`, которая безопасно обращается к элементу массива. Если `index` выходит за допустимые пределы `[0, size)`, функция должна вывести `{p["oom_msg"]}` и вернуть `{p["default_val"]}`. В противном случае — вывести `{p["ok_msg"]}` и вернуть значение `arr[index]`.

**Формат вывода:**
- При успехе: `{fmt_ok}`
- При ошибке: `{fmt_oom}`

**Уникальными значениями становятся:** `func_name`, `elem_type`, `ok_msg`, `oom_msg`, `elem_format`, `default_val`
`seed % 2 == 0`: func_name = "safe_get_int", elem_type = "int", ok_msg = "OK: ", oom_msg = "OOB: ", format = "%d\\n", default = "-1"
`seed % 2 == 1`: func_name = "safe_get_float", elem_type = "float", ok_msg = "Value: ", oom_msg = "Error: ", format = "%.2f\\n", default = "0.0f"

**Ввод:** Массив, размер и индекс передаются в функцию как аргументы.

**Пример решения:** (для seed=15)
```c
#include <stdio.h>

float safe_get_float(float *arr, int size, int index) {
    if (index < 0 || index >= size) {
        printf("Error: %.2f\n", 0.0f);
        return 0.0f;
    }
    printf("Value: %.2f\n", arr[index]);
    return arr[index];
}
```
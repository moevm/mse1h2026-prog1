### Тема: Передача функции как аргумент
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{func_name}({elem_type} *arr, int size, {elem_type} (*{callback_name})({elem_type}))`, которая применяет функцию-колбэк `{callback_name}` к каждому элементу массива `arr` и выводит результат в формате `{fmt_str}`.

**Формат вывода:** `{fmt_str}` (для каждого элемента, где `%d` или `%.1f` заменяется на результат `callback(arr[i])`)

**Уникальными значениями становятся:** `func_name`, `elem_type`, `callback_name`, `fmt_str`
`seed % 3 == 0`: func_name = "apply_transform", elem_type = "int", callback_name = "op", fmt_str = "Result[%d]: %d\n"
`seed % 3 == 1`: func_name = "process_values", elem_type = "float", callback_name = "transform", fmt_str = "Out[%d]: %.1f\n"
`seed % 3 == 2`: func_name = "map_array", elem_type = "int", callback_name = "mapper", fmt_str = "Val[%d] => %d\n"

**Ввод:** Массив, размер и функция-колбэк передаются в функцию как аргументы. 

**Пример решения:** (для seed=15)
```c
#include <stdio.h>

void apply_transform(int *arr, int size, int (*op)(int)) {
    for (int i = 0; i < size; i++) {
        int res = op(arr[i]); 
        printf("Result[%d]: %d\n", i, res);
    }
}
```



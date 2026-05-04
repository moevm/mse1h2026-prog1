### Тема: Многомерные массивы

**Сложность:** средняя
**Задание:**
Реализуйте функцию `[func_name](int matrix[][COLS], int rows, int target)`, которая принимает двумерный массив фиксированной ширины равной 4, количество строк `rows` и индекс `target`. Функция должна {operation}. 
Формат вывода:
{print_format}

**Уникальными значениями становятся:**
`func_name`, `operation`, `print_format`
`seed % 4 == 0`: func_name = "calc_row_sum", operation = "вычислить сумму target строки", print_format = "Sum row %d: %d\n"
`seed % 4 == 1`: func_name = "find_col_max", operation = "найти максимальный элемент в target столбце", print_format = "Max col %d: %d\n"
`seed % 4 == 2`: func_name = "calc_col_sum", operation = "вычислить сумму target столбца", print_format = "Sum col %d: %d\n"
`seed % 4 == 3`: func_name = "find_row_min", operation = "найти минимальный элемент в target строке", print_format = "Min row %d: %d\n"

Ввод: Массив инициализируется в `main()` явно. Параметры `rows` и `target` передаются в функцию. 

**Пример:** (для seed=13 генерируются `func_name` = "find_col_max", `target` = 2, `print_format` = "Max col %d: %d\n")

**Пример решения:**
```c
#include <stdio.h>
#define COLS 4

void find_col_max(int matrix[][COLS], int rows, int target) {
    if (target < 0 || target >= COLS) return;
    
    int max_val = matrix[0][target];
    for (int i = 1; i < rows; i++) {
        if (matrix[i][target] > max_val) {
            max_val = matrix[i][target];
        }
    }
    printf("Max col %d: %d\n", target, max_val);
}
```

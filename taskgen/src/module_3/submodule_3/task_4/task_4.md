### Тема: Массивы vs Указатели 
**Сложность:** средняя

**Задание:**
Реализуйте функцию `void {func_name}(int size)`, которая объявляет массив целых чисел `int arr[size]` и инициализирует его элементы по формуле: `arr[i] = {init_formula}`. Выведите все элементы массива, используя два способа доступа: через индекс `arr[i]` и через арифметику указателей.

**Формат вывода:** `{print_format}`

**Уникальными значениями становятся:** `func_name`, `init_formula`, `print_format`
`seed % 4 == 0`: func_name = "print_dual_arr", init_formula = "i * 2 + 1", print_format = "Idx[%d]: %d | Ptr: %d\n"
`seed % 4 == 1`: func_name = "show_idx_vs_ptr", init_formula = "i * i", print_format = "El[%d]: %d | Ptr: %d\n"
`seed % 4 == 2`: func_name = "array_dual_output", init_formula = "100 - i * 3", print_format = "Arr[%d]: %d | Ptr: %d\n"
`seed % 4 == 3`: func_name = "demo_access_modes", init_formula = "i + 10", print_format = "Out[%d]: %d | Ptr: %d\n"

**Ввод:** Размер массива `size` передаётся в функцию как аргумент из `main()`.

**Пример:** (для seed=15 генерируются `func_name` = "demo_access_modes", `init_formula` = "i + 10", `print_format` = "Out[%d]: %d | Ptr: %d\n")

**Пример решения:**
```c
#include <stdio.h>

void demo_access_modes(int size) {
    int arr[size];

    for (int i = 0; i < size; i++) {
        arr[i] = i + 10;
    }

    for (int i = 0; i < size; i++) {
        printf("Out[%d]: %d | Ptr: %d\n", i, arr[i], *(arr + i));
    }
}

Проверка осуществляется следующим образом: предпроверка - regex: использование арифметики указателей(разрешено: *(arr + i), *ptr++ и т.п.), сравнение строки вывода(обращение к элементу через arr[i] и через указатель); После компиляция и тесты с передачей в функцию разных размеров size;
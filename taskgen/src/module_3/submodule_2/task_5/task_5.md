### Тема: Разыменовывание

# Сложность: легкая

# Задание:
 Напишите функцию `{func_name}({type} *{p1}, {type} *{p2}, {type} *{p3})`, которая разыменовывает указатели `{p1}` и `{p2}`, вычисляет сумму значений и записывает результат по адресу `{p3}`. Внутри функции выведите полученное значение в формате `{fmt_str}`.

 - Уникальными значениями становятся:
`func_name`, `p1`, `p2`, `p3`, `fmt_str`

Правила генерации параметров (по seed):
seed % 3 == 0: func_name = "sum_via_ptrs", p1 = "a", p2 = "b", p3 = "res"
seed % 3 == 1: func_name = "add_pointed_vals", p1 = "lhs", p2 = "rhs", p3 = "out"
seed % 3 == 2: func_name = "compute_sum_ref", p1 = "val_x", p2 = "val_y", p3 = "target"
seed % 4 == 0: fmt_str = "Calculated: %d\n"
seed % 4 == 1: fmt_str = "Результат => %d\n"
seed % 4 == 2: fmt_str = "[Output] %d\n"
seed % 4 == 3: fmt_str = "Final value -> %d\n"

 - Ввод: Значения инициализируются в `main`, в функцию передаются их адреса (например, 10 и 7).
# Пример: 
 для seed=15 генерируются: func_name = "compute_sum_ref", p1 = "val_x", p2 = "val_y", p3 = "target", fmt_str = "Final value -> %d\n"
 
 - Ввод: *val_x = 10, *val_y = 7

 - Вывод:
  Final value -> 17

 - Пример решения:
```c
#include <stdio.h>

void compute_sum_ref(int *val_x, int *val_y, int *target) {
    *target = *val_x + *val_y;
    printf("Final value -> %d\n", *target);
}

int main(void) {
    int x = 10;
    int y = 7;
    int res = 0;
    
    compute_sum_ref(&x, &y, &res);
    return 0;
}
```
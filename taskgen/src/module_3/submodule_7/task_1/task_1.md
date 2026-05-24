### Тема: const int *, int * const, const int * const
**Сложность:** легкая

**Задание:**
Реализуйте функцию `{func_name}({data_type} {arg_name})`, которая внутри объявляет три указателя:
1. `{p1}` — указатель на константу
2. `{p2}` — константный указатель
3. `{p3}` — константный указатель на константу
Все три должны инициализироваться адресом `{arg_name}`.
Выведите разыменованные значения в формате:
`{fmt_str}`

**Уникальными значениями становятся:** `func_name`, `data_type`, `arg_name`, `p1`, `p2`, `p3`, `fmt_str`
`seed % 3 == 0`: func_name = "demo_const_ptrs", data_type = "int", arg_name = "val", p1 = "ro_ptr", p2 = "fix_ptr", p3 = "cc_ptr", fmt_str = "1: %d\n2: %d\n3: %d\n"
`seed % 3 == 1`: func_name = "demo_const_ptrs", data_type = "float", arg_name = "x", p1 = "f_ro", p2 = "f_fix", p3 = "f_cc", fmt_str = "1: %.2f\n2: %.2f\n3: %.2f\n"
`seed % 3 == 2`: func_name = "demo_const_ptrs", data_type = "char", arg_name = "c", p1 = "c_ro", p2 = "c_fix", p3 = "c_cc", fmt_str = "1: %c\n2: %c\n3: %c\n"

**Ввод:** Значение передаётся в функцию. 

**Пример:** (для seed=15 → `seed % 3 == 0`)
```c
#include <stdio.h>

void demo_const_ptrs(int val) {
    const int *ro_ptr = &val;
    int *const fix_ptr = &val;
    const int *const cc_ptr = &val;
    printf("1: %d\n2: %d\n3: %d\n", *ro_ptr, *fix_ptr, *cc_ptr);
}
```
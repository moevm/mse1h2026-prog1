### Тема: Таблицы функций (массивы указателей)
**Сложность:** средняя

**Задание:**
Реализуйте функцию `{func_name}(int op, {elem_type} a, {elem_type} b)`, которая выбирает и вызывает функцию из таблицы по индексу `op` и выводит результат в формате `{fmt_str}`.

**Таблица операций (индекс - функция):**
| `op` | Операция | Реализация |
|------|----------|-----------|
| 0 | Сложение | `a + b` |
| 1 | Вычитание | `a - b` |
| 2 | Умножение | `a * b` |
| 3 | Деление | `b != 0 ? a / b : 0` |

**Запрещено** использовать `switch`/`if-else` для выбора операции. Используйте массив указателей на функции.

**Формат вывода:** `{fmt_str}` (где `%d` или `%.1f` заменяется на результат операции)

**Уникальными значениями становятся:** `func_name`, `elem_type`, `fmt_str`
`seed % 3 == 0`: func_name = "calc_via_table", elem_type = "int", fmt_str = "Result: %d\n"
`seed % 3 == 1`: func_name = "execute_op", elem_type = "float", fmt_str = "Output: %.1f\n"
`seed % 3 == 2`: func_name = "dispatch", elem_type = "int", fmt_str = "=> %d\n"

**Ввод:** `op`, `a`, `b` передаются в функцию как аргументы.

**Пример решения:** (для seed=15)
```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int mul(int a, int b) { return a * b; }
int divide(int a, int b) { return b ? a / b : 0; }

void calc_via_table(int op, int a, int b) {
    int (*ops[])(int, int) = {add, sub, mul, divide};

    if (op >= 0 && op < 4) {
        printf("Result: %d\n", ops[op](a, b));
    }
}
```
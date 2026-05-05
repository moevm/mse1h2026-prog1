# Базовые типы данных

### Задание №8
- Сложность: сложная.

- Задание: напишите программу, которая выводит через пробел минимальное и максимальное значение типа `TYPENAME`, используя соответствующие макросы из `<limits.h>`. Программа должна содержать `#include <stdio.h>`, `#include <limits.h>` и функцию `main`. Спецификаторы формата для `printf` выберите так:  
 - для `char` и `short` - `%d`;  
 - для `int` - `%d`;  
 - для `long` - `%ld`;  
 - для `long long` - `%lld`.   

- Уникальными становятся значения `TYPENAME` (`char`, `short`, `int`, `long` или `long long`)

- Ввод: пуст. Вывод при `TYPE = long`:
```
#include <stdio.h>
#include <limits.h>

int main()
{
    printf("%ld %ld\n", LONG_MIN, LONG_MAX);
    return 0;
}
```
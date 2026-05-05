# Выражения и операторы

### Задание №11
- Сложность: легкая.

- Задание: ниже представлен фрагмент программы. Что он выведет? Запишите два числа через пробел.
```
signed int a = A;
int s = S;
unsigned int b = (unsigned int)a;
printf("%d %u", a >> s, b >> s);
``` 

- Уникальными становятся значения `A`, `S`.

- Ввод: пуст. Вывод при `A = -13`, `S = 2`:
```
#include <stdio.h>

int main()
{
    signed int a = -13;
    int s = 2;
    unsigned int b = (unsigned int)a;
    printf("%d %u", a >> s, b >> s);
    return 0;
}
```
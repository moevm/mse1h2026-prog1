# Управляющие конструкции

### Задание №4
- Сложность: средняя.

- Задание: напишите программу, которая использует оператор `switch`. Требования:  
  - Объявите переменную `val` заданного типа `type` и инициализируйте её значением `value`.  
  - С помощью `switch` обработайте три ситуации:  
    - если значение `val` отрицательное - выведите `"negative"`;  
    - если положительное - `"positive"`;  
    - если нулевое - `"zero"`.  
  - Программа должна содержать `#include <stdio.h>` и функцию `main`. 

- Уникальными становятся значения `value`, `type` (char, short, int, long).

- Ввод: пуст. Вывод при `value = -5`, `type = int`:
```
#include <stdio.h>

int main()
{
    int val = -5;
    switch ((val > 0) - (val < 0))
    {
        case -1: printf("negative"); break;
        case 0: printf("zero"); break;
        case 1: printf("positive"); break;
    }
    return 0;
}
```
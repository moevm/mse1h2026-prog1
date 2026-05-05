# Базовые типы данных

### Задание №10
- Сложность: сложная.

- Задание: напишите программу, которая использует заголовочный файл `<stdbool.h>`, объявляет переменную `flag` типа `bool`, присваивает ей значение `true`, если целое число `value` не меньше `const`, и `false` в противном случае, затем выводит строку `yes`, если `flag == true`, и `no` иначе. Программа должна содержать `#include <stdio.h>`, `#include <stdbool.h>` и функцию `main`.   

- Уникальными становятся значения `value`, `const`.

- Ввод: пуст. Вывод при `value = 13`, `const = 77`:
```
#include <stdio.h>
#include <stdbool.h>

int main()
{
    bool flag = 13 >= 77;
    if (flag == true)
        printf("yes");
    else
        printf("no");
    return 0;
}
```
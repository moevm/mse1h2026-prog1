# Управляющие конструкции

### Задание №2
- Cложность: легкая.
- Задание: напишите, что выведет программа:
```
    #include <stdio.h>
    
    int main()
    {
        int a = const1;
        int b = const2;
        int c = const3;
        if (a % 2 == bool) 
        {
            if (b / 7 > c) 
            {
                printf("%d\n", a + b);
            }
            else 
            {
                printf("%d\n", a - b);
            }
        }
        else 
        {
            printf("%d\n", a + c - b);
        }
        return 0;
    }
```

- Уникальными становится значения `const1, const2, const3, bool`. Значение `bool` - это `0 или 1`.

- Ввод: пуст. Пример вывода для `const1 = 13, const2 = 6, const3 = 2, bool = 0`: `9`.

# Структура программы в си

### Задание №5
- Cложность: легкая.
- Задание: что выведет программа при запуске?
```
    #include <stdio.h>

    int main()
    {
        int x = const1;
        double y = const2;
        char z = const3;
        printf(".x=%d, y =%f z= %c\n", x, y, z);
        return 0;
    }
```

- Уникальными становятся значения `const1`, `const2` и `const3`.

- Ввод: пуст. Пример вывода при `const1=13`, `const2=13.13`, `const3='S': `.x=13, y =13.130000 z= S` или `.x=13, y =13.130000 z= S\n`.

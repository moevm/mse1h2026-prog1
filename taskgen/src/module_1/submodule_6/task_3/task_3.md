# Управляющие конструкции

### Задание №3
- Cложность: средняя.
- Задание: напишите, что выведет программа:
```
#include <stdio.h>

int main() {
    int a = const1;
    int b = const2;
    int c = const3;

    switch (a % 3) {
        case 0:
            switch (b % 2) {
                case 0:
                    printf("%d\n", a + b + c);
                    break;
                case 1:
                    printf("%d\n", a - b + c);
                    break;
            }
            break;

        case 1:
            if (c > 5) {
                printf("%d\n", a * b);
            } else {
                printf("%d\n", a + b);
            }
            break;

        case 2:
            switch (c % 2) {
                case 0:
                    printf("%d\n", a * c);
                    break;
                default:
                    printf("%d\n", b * c);
                    break;
            }
            break;

        default:
            printf("%d\n", a + b + c);
    }

    return 0;
}
```

- Уникальными становится значения `const1, const2, const3`.

- Ввод: пуст. Пример вывода для `const1 = 7, const2 = 4, const3 = 3: `11`.

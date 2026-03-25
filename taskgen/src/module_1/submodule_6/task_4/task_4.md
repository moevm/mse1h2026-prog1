# Управляющие конструкции

### Задание №4
- Cложность: средняя.
- Задание: напишите, что выведет программа:
```
#include <stdio.h>

int main() {
    int a = const1;
    int b = const2;
    int c = const3;
    int d = const4;
    int result = 0;

    for (int i = a; i <= b; i++) {
        for (int j = c; j >= d; --j) {
            if (i > j) {
                result += i * j;
            } else {
                result -= i + j;
            }
        }
    }

    printf("%d\n", result);
    return 0;
}
```

- Уникальными становится значения `const1, const2, const3, const4`.

- Ввод: пуст. Пример вывода для `const1 = 4, const2 = 8, const3 = 13, const4 = 7`: `-489`.

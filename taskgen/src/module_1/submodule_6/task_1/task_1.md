# Управляющие конструкции

### Задание №1
- Cложность: легкая.
- Задание: напишите, что выведет программа:
    #include <stdio.h>
    
    int main()
    {
        int a = const1;
        while (a % const2) 
        {
            printf("%d ", operation(a));
        }
        return 0;
    }
- Уникальными становится значения const1, const2, operation(a).
const1 = (seed % 10 + 1) * const2 + const3
const2 = seed % 4 + 3
seed % 4 == 0: operation(a) = ++a
seed % 4 == 1: operation(a) = a++
seed % 4 == 2: operation(a) = --a
seed % 4 == 3: operation(a) = a--
seed % 4 == 0 or seed % 4 == 1: const3 = 1
seed % 4 == 2 or seed % 4 == 3: const3 = -1
- Ввод: пуст. Пример вывода для seed=13: const1 = 17, const2 = 4, operation(a) = a++, const3 = 1. Ответ: 17 18 19.

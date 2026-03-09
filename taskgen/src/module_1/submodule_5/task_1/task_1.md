# Выражения и операторы

### Задание №1
- Cложность: средняя.
- Задание: чему равно значение переменной c после вычисления данного фрагмента кода?
    int a = const1;
    int b = const2;
    int c = operation1 a operation2 b operation3;
    c operation4 = b operatin5 a;
- Уникальными становятся значения const1, const2, operation1-5.
const1 = seed
const2 = seed / 7 + seed % 7
seed % 2 == 0: operation1 = ++
seed % 2 == 1: operation1 = --
seed % 4 == 0: operation2 = +
seed % 4 == 1: operation2 = -
seed % 4 == 2: operation2 = *
seed % 4 == 3: operation2 = /
(seed / 5) % 2 == 0: operation3 = ++
(seed / 5) % 2 == 1: operation3 = --
(seed / 3) % 4 == 0: operation4 = +
(seed / 3) % 4 == 1: operation4 = -
(seed / 3) % 4 == 2: operation4 = *
(seed / 3) % 4 == 3: operation4 = /
(seed / 5) % 4 == 0: operation5 = +
(seed / 5) % 4 == 1: operation5 = -
(seed / 5) % 4 == 2: operation5 = *
(seed / 5) % 4 == 3: operation5 = /
- Ввод: пуст. Пример вывода для seed=13: const1 = 13, const2 = 7, operation1 = --, operation2 = -, operation3 = ++, operation4 = +, operation5 = *. Ответ: c = 101.
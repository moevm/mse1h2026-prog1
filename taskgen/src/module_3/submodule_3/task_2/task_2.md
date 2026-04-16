### Тема: "Инициализация через указатель"

**Сложность:** средняя

**Задание:**
Объявите массив `arr` размером `{size}` элементов типа `{type}`.
Объявите указатель `{type} *ptr` на начало массива.
Заполните массив в цикле по формуле: `{formula}`.
Используйте доступ к элементам массива через указатель.
Выведите значение элемента по индексу `{output_index}`.

**Формат вывода:**
`Value: X`

**Уникальными значениями становятся:**
`size`, `type`, `formula`, `output_index`

seed % 2 == 0: size = (seed % 15); type = int; formula = i * (seed % 1000);output_index = (seed % 15)

seed % 2 == 1: size=(seed % 15); type=float; formula=((seed % 1000) * 0.5) + i; output_index=(seed % 15)

## Ввод:
Массив задаётся в коде.

## Пример для seed=15:
type 15 % 3 = 0 `int` 
formula, output_index 15 % 4 = 3 `i + 10`, индекс `2` 
size 15 % 5 = 0 `10` 

**Пример решения:**
```c
#include <stdio.h>

int main() {
    int arr[10];
    int *ptr = arr;
    int i;
    
    for (i = 0; i < 10; i++) {
        *(ptr + i) = i + 10;
    }
    
    printf("Value: %d\n", *(ptr + 2));
    return 0;
}
```

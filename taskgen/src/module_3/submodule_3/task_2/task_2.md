# Тема: "Инициализация через указатель"
**Сложность:** средняя

## Задание:
Объявите массив `arr` размером `{size}` элементов типа `{type}`.
Объявите указатель `{pointer_type} *ptr` на начало массива.
Заполните массив в цикле по формуле: `{formula}`.
Используйте доступ к элементам массива через указатель.
Выведите значение элемента по индексу `{output_index}`.

**Формат вывода:**
`Value: X`

## Уникальными значениями становятся: `type`, `size`, `formula`, `output_index`

seed % 3 == 0: type = int, pointer_type = int 
seed % 3 == 1: type = float, pointer_type = float 
seed % 3 == 2: type = char, pointer_type = char 

seed % 4 == 0: formula = "i + 3", output_index = 5 
seed % 4 == 1: formula = "i * 2", output_index = 3 
seed % 4 == 2: formula = "i - 1", output_index = 7 
seed % 4 == 3: formula = "i + 10", output_index = 2 

seed % 5 == 0: size = 10 
seed % 5 == 1: size = 15 
seed % 5 == 2: size = 20 
seed % 5 == 3: size = 12 
seed % 5 == 4: size = 8 

## Ввод:
Массив задаётся в коде (генерируется автоматически). Ввод пустой.

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

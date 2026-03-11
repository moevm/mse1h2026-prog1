# Указатели

### Раздел "Арифметика указателей"
### Задание №1
- Cложность: легкая.
- Задание: Объявите переменную типа type, инициализируйте ее значением value, передайте ее адрес в указатель. Приметите операцию operation к переменной используя арифметику указателей. 

Результат вывести в формате: 
Value before: var_before_modification 
Value after: var_after_modification

- Уникальными значениями становятся: `type`, `value`, `operation`.

seed % 2 == 0: type = int, value = seed + 15
seed % 2 == 1: type = float, value = seed/2
seed % 4 == 0: operation = "инкремента"
seed % 4 == 1: operation = "увеличения на 5.87"
seed % 4 == 2: operation = "декремента"
seed % 4 == 3: operation = "уменьшения на 4.2"

- Ввод: пустой.
- Пример для seed=15:
  - type = float (15 % 2 = 1)
  - value = 7.5
  - operation = "уменьшения на 4.2"
  
Ожидаемый вывод:
Value before: 7.5
Value after: 3.3
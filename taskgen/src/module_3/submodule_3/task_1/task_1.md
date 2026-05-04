#  Тема: Объявление

**Сложность:** легкая

**Задание:** Объявите массив с именем {name_arr} размером {size} элементов типа {type}.

**Уникальными значениями становятся:** name_arr, type, size

 seed % 3 == 0: name_arr = int_arr; type = int; size = (seed % 15 * (seed % 100) + 2)
 seed % 3 == 1: name_arr = float_arr; type = float; size = (seed % 15 * (seed % 100) + 2)
 seed % 3 == 2: name_arr = char_arr; type = char; size = (seed % 15 * (seed % 100) + 2)

 Ввод: пустой
 
 **Пример для seed=15:**
 Ожидаемый вывод: int int_arr[2]






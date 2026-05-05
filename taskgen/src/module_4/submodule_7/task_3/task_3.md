# Блок для самых умных: битовые поля


### Задание №3
- **Сложность:** сложно
- **Формулировка:**  
  Вам дано объявление структуры с битовыми полями (соответствует вашему варианту).  
  Напишите функцию `pack` и функцию `unpack`:
  - `pack` принимает отдельные значения полей и возвращает заполненную структуру.
  - `unpack` принимает структуру и выводит значения всех её полей в стандартный вывод в порядке, указанном в таблице.

  Писать `main()` не нужно.

- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | Объявление структуры (дано в условии) | Сигнатура `pack` | Сигнатура `unpack` | Порядок вывода в `unpack` | Формат вывода |
  |------------|---------------------------------------|------------------|--------------------|---------------------------|---------------|
  | 0 | `typedef struct { unsigned int r : 5; unsigned int g : 5; unsigned int b : 5; unsigned int a : 1; } Color16;` | `Color16 pack(unsigned r, unsigned g, unsigned b, unsigned a)` | `void unpack(Color16 c)` | `r`, `g`, `b`, `a` | `%u %u %u %u` |
  | 1 | `typedef struct { unsigned int day : 5; unsigned int month : 4; unsigned int year : 7; } Date;` | `Date pack(unsigned day, unsigned month, unsigned year)` | `void unpack(Date d)` | `day`, `month`, `year` | `%u %u %u` |
  | 2 | `typedef struct { unsigned int hour : 5; unsigned int min : 6; unsigned int sec : 6; } Time;` | `Time pack(unsigned hour, unsigned min, unsigned sec)` | `void unpack(Time t)` | `hour`, `min`, `sec` | `%u %u %u` |
  | 3 | `typedef struct { unsigned int x : 6; unsigned int y : 6; unsigned int flags : 4; } Tile;` | `Tile pack(unsigned x, unsigned y, unsigned flags)` | `void unpack(Tile t)` | `x`, `y`, `flags` | `%u %u %u` |


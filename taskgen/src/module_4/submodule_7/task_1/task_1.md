# Блок для самых умных: структура в структуре


### Задание №1
- **Сложность:** сложно
- **Формулировка:**  
  Вам даны объявления двух вложенных структур (соответствует вашему варианту).  
  Напишите функцию `find_best`, которая:
  1. Принимает массив структур и его размер.
  2. Находит элемент массива, у которого значение поля вложенной структуры `inner_field` максимально.
  3. Возвращает указатель на этот элемент.

  Писать `main()` не нужно.

- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | Объявление структур (дано в условии) | Сигнатура `find_best` | `inner_field` | Формат вывода поля внешней структуры |
  |------------|--------------------------------------|-----------------------|---------------|--------------------------------------|
  | 0 | `typedef struct { float lat; float lon; } Coords;`<br>`typedef struct { char name[32]; Coords location; int rating; } Place;` | `Place *find_best(Place *arr, int n)` | `location.lat` (`float`) | `%.2f` (вывести `location.lat` найденного) |
  | 1 | `typedef struct { int year; int month; } Date;`<br>`typedef struct { char title[64]; Date published; int sales; } Book;` | `Book *find_best(Book *arr, int n)` | `published.year` (`int`) | `%d` (вывести `published.year` найденного) |
  | 2 | `typedef struct { double x; double y; } Point;`<br>`typedef struct { Point center; double radius; char color[16]; } Circle;` | `Circle *find_best(Circle *arr, int n)` | `center.x` (`double`) | `%.1f` (вывести `center.x` найденного) |
  | 3 | `typedef struct { int hours; int minutes; } Time;`<br>`typedef struct { char event[32]; Time start; int duration; } Schedule;` | `Schedule *find_best(Schedule *arr, int n)` | `start.hours` (`int`) | `%d` (вывести `start.hours` найденного) |

  Программа считывает из стандартного ввода `n = 3` объектов внешней структуры, вызывает `find_best` и выводит значение `inner_field` найденного элемента.

- **Замечание:**  
  Функция `find_best` должна возвращать именно **указатель** на элемент массива, а не копию.  
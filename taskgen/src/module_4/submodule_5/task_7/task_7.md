# stdlib: qsort

### Задание №7
- **Сложность:** средне
- **Формулировка:**  
  Вам дано объявление структуры и заполненный массив объектов (соответствуют вашему варианту).  
  Напишите:
  1. Функцию-компаратор `cmp_<TypeName>` для использования с `qsort`.
  2. Функцию `sort_arr`, которая принимает массив структур и его длину и сортирует его по заданному полю в заданном направлении с помощью `qsort`.

  Писать `main` не нужно — только компаратор и тело функции `sort_arr`.  

- **Параметры задания:**  
  Вариант определяется значением `seed`.

  - **Направление сортировки** `dir = seed % 2`:
    - `0` — по возрастанию
    - `1` — по убыванию

  | `seed % 5` | Тип структуры | Объявление структуры | Поле для сортировки (тип) | Сигнатуры функций |
  |------------|---------------|----------------------|---------------------------|-------------------|
  | 0 | `Student` | `typedef struct { char name[32]; int age; float gpa; } Student;` | `age` (`int`) | `int cmp_Student(const void*, const void*)`<br>`void sort_arr(Student arr[], int n)` |
  | 1 | `Book` | `typedef struct { char title[64]; int pages; double price; } Book;` | `pages` (`int`) | `int cmp_Book(const void*, const void*)`<br>`void sort_arr(Book arr[], int n)` |
  | 2 | `Point2D` | `typedef struct { double x; double y; int label; } Point2D;` | `label` (`int`) | `int cmp_Point2D(const void*, const void*)`<br>`void sort_arr(Point2D arr[], int n)` |
  | 3 | `Rectangle` | `typedef struct { double width; double height; char color[16]; } Rectangle;` | `width` (`double`) | `int cmp_Rectangle(const void*, const void*)`<br>`void sort_arr(Rectangle arr[], int n)` |
  | 4 | `Employee` | `typedef struct { char name[32]; int id; double salary; } Employee;` | `salary` (`double`) | `int cmp_Employee(const void*, const void*)`<br>`void sort_arr(Employee arr[], int n)` |

- **Пример для `seed = 7` (`seed % 5 = 2` → `Point2D`, `dir = 1` → убывание по `label`):**

  Функция вызвана как `sort_arr(arr, 3)`, где массив содержит:
  ```
  { .x=1.0, .y=2.0, .label=10 }
  { .x=3.5, .y=4.5, .label=20 }
  { .x=0.0, .y=1.0, .label=15 }
  ```
  Ожидаемый порядок после сортировки:
  ```
  { .x=3.5, .y=4.5, .label=20 }
  { .x=0.0, .y=1.0, .label=15 }
  { .x=1.0, .y=2.0, .label=10 }
  ```
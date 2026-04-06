# stdlib: bsearch

### Задание №8
- **Сложность:** средне
- **Формулировка:**  
  Вам дан массив структур указанного типа, отсортированный по полю **key_field** в порядке возрастания.  
  Напишите:
  1. Функцию-компаратор `cmp_<TypeName>` для использования с `bsearch`.
  2. Функцию `find_in_arr`, которая принимает отсортированный массив структур, его длину и значение ключа, и возвращает **указатель** на найденный элемент или `NULL`, если элемент не найден.

  Писать `main` не нужно — только компаратор и тело функции `find_in_arr`.

- **Параметры задания:**  
  Вариант определяется значением `seed % 5`:

  | `seed % 5` | Тип структуры | Объявление структуры | Ключевое поле (тип) | Сигнатуры функций |
  |------------|---------------|----------------------|---------------------|-------------------|
  | 0 | `Student` | `typedef struct { char name[32]; int age; float gpa; } Student;` | `age` (`int`) | `int cmp_Student(const void*, const void*)`<br>`Student *find_in_arr(Student arr[], int n, int key)` |
  | 1 | `Book` | `typedef struct { char title[64]; int pages; double price; } Book;` | `pages` (`int`) | `int cmp_Book(const void*, const void*)`<br>`Book *find_in_arr(Book arr[], int n, int key)` |
  | 2 | `Point2D` | `typedef struct { double x; double y; int label; } Point2D;` | `label` (`int`) | `int cmp_Point2D(const void*, const void*)`<br>`Point2D *find_in_arr(Point2D arr[], int n, int key)` |
  | 3 | `Rectangle` | `typedef struct { double width; double height; char color[16]; } Rectangle;` | `width` (`double`) | `int cmp_Rectangle(const void*, const void*)`<br>`Rectangle *find_in_arr(Rectangle arr[], int n, double key)` |
  | 4 | `Employee` | `typedef struct { char name[32]; int id; double salary; } Employee;` | `salary` (`double`) | `int cmp_Employee(const void*, const void*)`<br>`Employee *find_in_arr(Employee arr[], int n, double key)` |

- **Пример для `seed = 2` (`seed % 5 = 2` → `Point2D`, ключевое поле `label`):**

  Функция вызвана как `find_in_arr(arr, 4, 20)`, где массив содержит:
  ```
  { .x=1.0, .y=2.0, .label=10 }
  { .x=2.0, .y=3.0, .label=20 }
  { .x=3.0, .y=4.0, .label=30 }
  { .x=4.0, .y=5.0, .label=40 }
  ```
  - `find_in_arr(arr, 4, 20)` → указатель на `{ .x=2.0, .y=3.0, .label=20 }`
  - `find_in_arr(arr, 4, 99)` → `NULL`

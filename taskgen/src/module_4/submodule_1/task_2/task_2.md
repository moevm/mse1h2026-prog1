# Struct: инициализация переменной

### Задание №2
- **Сложность:** легко  
- **Формулировка:**  
  Вам дано объявление структуры (оно соответствует вашему варианту).  
  Напишите функцию `init_<TypeName>`, которая:
  1. Принимает значения всех полей структуры в качестве аргументов.
  2. Возвращает переменную указанного типа, инициализированную **одним выражением** внутри `return`, используя инициализацию с указанием полей (**designated initializer**).

  Подключать заголовочные файлы не требуется.  
  Писать `main`, `typedef` и вызов функции не нужно — только тело функции.

  **Сигнатура функции задана и соответствует вашему варианту** (см. таблицу).  
  Функция будет вызвана автоматически; результат будет выведен тестирующей системой.

- **Параметры задания:**  
  Вариант задания определяется значением `seed` (вычисляется как хеш от ФИО студента).

  | `seed % 5` | Тип структуры | Объявление структуры (дано в условии) | Сигнатура функции |
  |------------|---------------|---------------------------------------|-------------------|
  | 0 | `Student` | `typedef struct { char name[32]; int age; float gpa; } Student;` | `Student init_Student(const char *name, int age, float gpa)` |
  | 1 | `Book` | `typedef struct { char title[64]; int pages; double price; } Book;` | `Book init_Book(const char *title, int pages, double price)` |
  | 2 | `Point2D` | `typedef struct { double x; double y; int label; } Point2D;` | `Point2D init_Point2D(double x, double y, int label)` |
  | 3 | `Rectangle` | `typedef struct { double width; double height; char color[16]; } Rectangle;` | `Rectangle init_Rectangle(double width, double height, const char *color)` |
  | 4 | `Employee` | `typedef struct { char name[32]; int id; double salary; } Employee;` | `Employee init_Employee(const char *name, int id, double salary)` |

  **Генерация значений, с которыми будет вызвана функция, на основе `seed`:**
  - `string_val` — выбирается из набора (`"Alice"`, `"Bob"`, `"Charlie"`, `"Diana"`, `"Eve"`, `"Frank"`, `"Grace"`, `"Hank"`, `"Ivy"`, `"Jack"`, `"Kevin"`, `"Laura"`, `"Mike"`, `"Nina"`, `"Oscar"`, `"Paula"`, `"Quentin"`, `"Rose"`, `"Steve"`, `"Tina"`, `"Ulysses"`, `"Vera"`, `"Will"`, `"Xena"`, `"Yves"`, `"Zoe"`). Индекс: `seed % 26`.
  - `int_val = 10 + (seed % 90)`
  - `float_val = (seed % 20) / 10.0f + 2.0f`
  - `double_val = (seed % 100) / 10.0 + 1.0`
  - Для структур с двумя `double`-полями:
    - `double_val1 = (seed % 32) / 10.0 + 1.0`
    - `double_val2 = (seed % 100) / 10.0 + 1.0`

- **Пример для `seed = 42` (`seed % 5 = 2` → `Point2D`):**

  Функция будет вызвана как `init_Point2D(3.2, 5.2, 52)`.  
  Ожидаемый результат: функция возвращает `Point2D` с полями `x = 3.2`, `y = 5.2`, `label = 52`.

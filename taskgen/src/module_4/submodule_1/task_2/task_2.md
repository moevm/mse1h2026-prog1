# Struct: инициализация переменной

### Задание №2
- **Сложность:** легко  
- **Формулировка:**  
  Вам дано объявление структуры (оно соответствует вашему варианту).  
  Напишите программу `main`, которая:
  1. Объявляет переменную с именем `obj` указанного типа.
  2. Инициализирует её **одним выражением при объявлении**, используя инициализацию с указанием полей, значениями из таблицы ниже.
  3. Выводит все поля структуры через пробел в одну строку в формате, указанном в варианте.

  Подключать дополнительные заголовочные файлы (кроме `stdio.h`) не требуется.
  Писать typedef структуры не нужно — только `main()`.

- **Параметры задания:**  
  Вариант задания определяется значением `seed` (вычисляется как хеш от ФИО студента).  
  Каждому варианту соответствует своя структура, набор полей, значения для инициализации и формат вывода.

  | `seed % 5` | Имя структуры | Объявление структуры (дано в условии) | Поля для инициализации | Формат вывода |
  |------------|---------------|---------------------------------------|------------------------|---------------|
  | 0 | `Student` | `typedef struct { char name[32]; int age; float gpa; } Student;` | `name = "<string>"`, `age = <int_val>`, `gpa = <float_val>` | `%s %d %.1f` |
  | 1 | `Book` | `typedef struct { char title[64]; int pages; double price; } Book;` | `title = "<string>"`, `pages = <int_val>`, `price = <double_val>` | `%s %d %.2f` |
  | 2 | `Point2D` | `typedef struct { double x; double y; int label; } Point2D;` | `x = <double_val1>`, `y = <double_val2>`, `label = <int_val>` | `%.1f %.1f %d` |
  | 3 | `Rectangle` | `typedef struct { double width; double height; char color[16]; } Rectangle;` | `width = <double_val1>`, `height = <double_val2>`, `color = "<string>"` | `%.1f %.1f %s` |
  | 4 | `Employee` | `typedef struct { char name[32]; int id; double salary; } Employee;` | `name = "<string>"`, `id = <int_val>`, `salary = <double_val>` | `%s %d %.2f` |

  **Генерация конкретных значений на основе `seed`:**
  - `string_val` — выбирается из предопределённого набора (`"Alice"`, `"Bob"`, `"Charlie"`, `"Diana"`, `"Eve"`, `"Frank"`, `"Grace"`, `"Hank"`, `"Ivy"`, `"Jack"`, `"Kevin"`, `"Laura"`, `"Mike"`, `"Nina"`, `"Oscar"`, `"Paula"`, `"Quentin"`, `"Rose"`, `"Steve"`, `"Tina"`, `"Ulysses"`, `"Vera"`, `"Will"`, `"Xena"`, `"Yves"`, `"Zoe"`). Индекс в массиве вычисляется как `seed % (количество имён)`.
  - `int_val = 10 + (seed % 90)`
  - `float_val = (seed % 20) / 10.0 + 2.0`
  - `double_val = (seed % 100) / 10.0 + 1.0`
  - Если структура содержит два `double` поля, значения вычисляются отдельно:
    - `double_val1 = (seed % 32) / 10.0 + 1.0`
    - `double_val2 = (seed % 100) / 10.0 + 1.0`

- **Пример вывода для `seed = 42` (`seed % 5 = 2` → `Point2D`):**  
  `3.2 5.2 52`
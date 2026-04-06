# Typedef: typedef struct

### Задание №2
- **Сложность:** легко
- **Формулировка:**  
  Объявите структурный тип `<Name>` так, чтобы переменную можно было объявить как `<Name> x;` без ключевого слова `struct`.


- **Параметры задания:**  
  Вариант определяется значением `seed % 5`:

  | `seed % 5` | Имя типа | Поля |
  |------------|----------|------|
  | 0 | `Student` | `char name[32]`, `int age`, `float gpa` |
  | 1 | `Book` | `char title[64]`, `int pages`, `double price` |
  | 2 | `Point2D` | `double x`, `double y`, `int label` |
  | 3 | `Rectangle` | `double width`, `double height`, `char color[16]` |
  | 4 | `Employee` | `char name[32]`, `int id`, `double salary` |

- **Пример для `seed = 10` (`seed % 5 = 0` → `Student`):**
  ```c
  typedef struct { char name; int age; float gpa; } Student;
  ```
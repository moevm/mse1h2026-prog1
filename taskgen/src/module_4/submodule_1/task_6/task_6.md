# Struct: оптимальный порядок полей

### Задание №6
- **Сложность:** средне
- **Формулировка:**  
  Вам дан набор полей структуры (определяется вашим вариантом).  
  Ниже представлены четыре варианта объявления — порядок полей разный.  
  Определите, какой вариант даёт **наименьший** `sizeof(struct S)` на платформе x86-64. Введите букву варианта и его `sizeof` через пробел.

  Введите ответ в формате: `<буква> <sizeof>`  
  Например: `A 16`

- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | Набор полей | Оптимальный порядок (A) | Варианты с неправильными порядками (B, C, D) |
  |------------|-------------|-------------------------|-----------------------------------------|
  | 0 | `char, double, char, int` | `double; int; char; char` | `char; double; int; char`<br>`char; char; int; double`<br>`int; char; double; char` |
  | 1 | `char, int, short, double` | `double; int; short; char` | `char; int; short; double`<br>`char; short; int; double`<br>`int; char; double; short` |
  | 2 | `int, char, short, long long` | `long long; int; short; char` | `int; char; short; long long`<br>`char; int; long long; short`<br>`short; int; char; long long` |
  | 3 | `char, int, char, short, double` | `double; int; short; char; char` | `char; int; char; short; double`<br>`char; char; int; double; short`<br>`int; char; short; double; char` |


- **Замечание:**  
  Буквы вариантов (A, B, C, D) перемешиваются для каждого студента в зависимости от `seed` — правильный порядок не всегда соответствует букве A.
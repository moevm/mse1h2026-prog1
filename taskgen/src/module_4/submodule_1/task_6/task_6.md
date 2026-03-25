# Struct: оптимальный порядок полей

### Задание №6
- **Сложность:** средне  
- **Формулировка:**  
  Вам нужно объявить структуру, содержащую набор полей (указан ниже).  
  Ниже представлены четыре варианта объявления — порядок полей разный.  
  Выберите вариант с **наименьшим** `sizeof(struct S)` на платформе x86-64. 

- **Параметры задания:**  
  Набор полей определяется значением `seed % 4`.  
  Правильный вариант всегда **A** (в исходной таблице), но перед показом студенту метки вариантов перемешиваются.  

   | `seed % 4` | Набор полей | Оптимальный порядок (A) | Варианты с неправильными порядками (B, C, D) |
  |------------|-------------|-------------------------|-----------------------------------------|
  | 0 | `char, double, char, int` | `double; int; char; char` | `char; double; int; char`<br>`char; char; int; double`<br>`int; char; double; char` |
  | 1 | `char, int, short, double` | `double; int; short; char` | `char; int; short; double`<br>`char; short; int; double`<br>`int; char; double; short` |
  | 2 | `int, char, short, long long` | `long long; int; short; char` | `int; char; short; long long`<br>`char; int; long long; short`<br>`short; int; char; long long` |
  | 3 | `char, int, char, short, double` | `double; int; short; char; char` | `char; int; char; short; double`<br>`char; char; int; double; short`<br>`int; char; short; double; char` |
# Enum: явное задание значений

### Задание №2
- **Сложность:** средне  
- **Формулировка:**  
  Объявите перечисление (соответствует вашему варианту) с явно заданными значениями членов.  
  Напишите функцию `combine_<EnumName>`, которая:
  1. Принимает два значения типа соответствующего `enum`.
  2. Возвращает результат операции `op` над их числовыми значениями (тип результата — `int`).

  Подключать заголовочные файлы не требуется.  
  Писать `main` не нужно — только объявление `enum` и тело функции.  
  **Сигнатура функции и операция `op` заданы и соответствуют вашему варианту** (см. таблицу).  

- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | Имя enum | Члены и явные значения | Операция `op` | Сигнатура функции |
  |------------|----------|------------------------|---------------|-------------------|
  | 0 | `HttpStatus` | `OK=200, FORBIDDEN=403, NOT_FOUND=404, SERVER_ERROR=500` | сумма двух значений | `int combine_HttpStatus(enum HttpStatus a, enum HttpStatus b)` |
  | 1 | `Priority` | `LOW=1, MEDIUM=5, HIGH=10, CRITICAL=100` | максимум из двух значений | `int combine_Priority(enum Priority a, enum Priority b)` |
  | 2 | `Permission` | `NONE=0, EXEC=1, WRITE=2, READ=4` | побитовое ИЛИ двух значений | `int combine_Permission(enum Permission a, enum Permission b)` |
  | 3 | `LogLevel` | `DEBUG=10, INFO=20, WARN=30, ERROR=40` | разность двух значений (`a - b`) | `int combine_LogLevel(enum LogLevel a, enum LogLevel b)` |

- **Пример для `seed = 3` (`seed % 4 = 3` → `LogLevel`, op = разность):**

  - `combine_LogLevel(WARN, INFO)` → `10` (т.к. `30 - 20 = 10`)
  - `combine_LogLevel(DEBUG, ERROR)` → `-30` (т.к. `10 - 40 = -30`)

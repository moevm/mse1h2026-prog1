# stdlib: fwrite


### Задание №5
- **Сложность:** средне
- **Формулировка:**  
  Вам дана сигнатура функции (соответствует вашему варианту).  
  Напишите функцию `write_items`, которая:
  1. Открывает файл с именем `filename` в бинарном режиме для записи.
  2. Записывает `count` элементов из массива `buf` в файл.
  3. Закрывает файл.
  4. Возвращает фактическое количество записанных элементов.  
  Если файл не удалось открыть — возвращает `0`.

  Писать `main()` не нужно.


- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | Сигнатура функции | `count` |
  |------------|-------------------|-----------------|
  | 0 | `int write_items(const char *filename, const int *buf, int count)` | 5 |
  | 1 | `int write_items(const char *filename, const double *buf, int count)` | 3 |
  | 2 | `int write_items(const char *filename, const float *buf, int count)` | 8 |
  | 3 | `int write_items(const char *filename, const short *buf, int count)` | 6 |


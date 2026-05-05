# stdlib: atexit


### Задание №10
- **Сложность:** легко
- **Формулировка:**  
  Дан фрагмент программы (соответствует вашему варианту).  
  Введите имена функций через пробел в том порядке, в котором они
  будут вызваны при завершении программы.

- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | Фрагмент | Ответ |
  |------------|----------|-------|
  | 0 | `atexit(A); atexit(B); atexit(C);` | `C B A` |
  | 1 | `atexit(log); atexit(cleanup); atexit(notify);` | `notify cleanup log` |
  | 2 | `atexit(free_mem); atexit(save_log);` | `save_log free_mem` |
  | 3 | `atexit(X); atexit(Y); atexit(Z); atexit(W);` | `W Z Y X` |
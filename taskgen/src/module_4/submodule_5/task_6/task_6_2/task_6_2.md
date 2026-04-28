# stdlib: бинарный режим


### Задание №6.2
- **Сложность:** средне
- **Формулировка:**  
  Напишите **две функции**:

  1. `void write_bin(const char *filename, value_type value)` — открывает файл
     `filename` в бинарном режиме для записи (`"wb"`), записывает одно значение
     через `fwrite`, закрывает файл.

  2. `value_type read_bin(const char *filename)` — открывает файл `filename`
     в бинарном режиме для чтения (`"rb"`), считывает одно значение через
     `fread`, закрывает файл и возвращает считанное значение.

  Если файл не удалось открыть — функция чтения возвращает `0` (или `0.0`).  
  Писать `main()` не нужно.


- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | `value_type` | Сигнатуры |
  |------------|-------------|-----------|
  | 0 | `int` | `void write_bin(const char*, int)`<br>`int read_bin(const char*)` |
  | 1 | `double` | `void write_bin(const char*, double)`<br>`double read_bin(const char*)` |
  | 2 | `long` | `void write_bin(const char*, long)`<br>`long read_bin(const char*)` |
  | 3 | `float` | `void write_bin(const char*, float)`<br>`float read_bin(const char*)` |



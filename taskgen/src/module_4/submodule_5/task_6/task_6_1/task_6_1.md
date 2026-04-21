# stdlib: текстовый режим


### Задание №6.1
- **Сложность:** средне
- **Формулировка:**  
  Напишите **две функции**:

  1. `void write_text(const char *filename, value_type value)` — открывает файл
     `filename` в текстовом режиме для записи, записывает значение
     в формате `format` через `fprintf`, закрывает файл.

  2. `value_type read_text(const char *filename)` — открывает файл `filename`
     в текстовом режиме для чтения, считывает одно значение через
     `fscanf`, закрывает файл и возвращает считанное значение.

  Если файл не удалось открыть — функция чтения возвращает `0` (или `0.0`).  
  Писать `main()` не нужно.


- **Параметры задания:**  
  Вариант определяется значением `seed % 4`:

  | `seed % 4` | `value_type` | `format` (для `fprintf`/`fscanf`) | Сигнатуры |
  |------------|-------------|-----------------------------------|-----------|
  | 0 | `int` | `"%d"` | `void write_text(const char*, int)`<br>`int read_text(const char*)` |
  | 1 | `double` | `"%.6f"` | `void write_text(const char*, double)`<br>`double read_text(const char*)` |
  | 2 | `long` | `"%ld"` | `void write_text(const char*, long)`<br>`long read_text(const char*)` |
  | 3 | `float` | `"%.4f"` | `void write_text(const char*, float)`<br>`float read_text(const char*)` |

### Запуск генерации задания

Для генерации задания используется точка входа `taskgen/main.py`.

Общая команда:

```bash
python <путь до точки входа main.py> <module_2.submodule_i.task_j> --mode init --seed <число>
```

Параметры:

- <module_2.submodule_i.task_j> - путь до папки с заданием, i - номер подмодуля, j - номер задания;
- --mode init - режим генерации условия;
- --seed <число> - число, определяющее вариант задания (seed).

Пример команды (если запускать из родительской папки mse1h2026-prog1):

```bash
python taskgen/main.py module_2.submodule_1.task_1 --mode init --seed 10
```

Пример вывода:
```bash
Какой флаг gcc останавливает сборку после этапа ассемблирования?
```

### Запуск проверки решения

Общая команда:

```bash
python <путь до точки входа main.py> <module_2.submodule_i.task_j> --mode check --seed <число> --solution="<ответ>"
```

Параметры:

- <module_2.submodule_i.task_j> - путь до папки с заданием, i - номер подмодуля, j - номер задания;
- --mode check - режим проверки решения;
- --seed <число> - число, определяющее вариант задания (seed);
- --solution="<ответ>" - ответ студента (строка), который проверяется.

Пример команды (если запускать из родительской папки mse1h2026-prog1):

```bash
python taskgen/main.py module_2.submodule_1.task_1 --mode check --seed 10 --solution="-c"
```

Пример вывода:
```bash
Passed: True
OK: Заглушка.
```
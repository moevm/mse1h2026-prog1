# Подмодуль 1: Структуры в C

Подмодуль содержит 7 заданий по работе со структурами. 

## Темы заданий

| № | Название | Тема |
|---|----------|------|
| 1 | Struct: объявление структуры | Синтаксис `struct` и `typedef` |
| 2 | Struct: инициализация переменной | Инициализация с указанием полей |
| 3 | Struct: доступ к полям | Чтение данных, вычисление максимума/суммы/минимума |
| 4 | Struct: выравнивание (alignment) | Смещение поля с учётом выравнивания |
| 5 | Struct: выравнивание (padding) | Количество байт выравнивания |
| 6 | Struct: оптимальный порядок полей | Выбор порядка полей для минимального размера |
| 7 | Struct: sizeof(struct) | Расчёт размера структуры |

## Запуск автоматизированных заданий

```bash
# Генерация задания и проверка решения
python taskgen/main.py struct_decl --mode init --seed 42 --solution solution.c

# Проверка только решения
python taskgen/main.py struct_decl --mode check --seed 42 --solution solution.c

# Аналогично для заданий 2 и 3
python taskgen/main.py struct_init --mode init --seed 42 --solution solution.c
python taskgen/main.py struct_access --mode init --seed 42 --solution solution.c
```
**Опции:**
- `-s, --seed` – вариант задания (целое число)
- `--solution` – путь к файлу с решением
- `--mode` – `init` (сгенерировать и проверить), `check` (только проверка), `dry-run` (только генерация)
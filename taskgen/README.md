# Модуль

## Зависимости

Платформа `docker`.

## Сборка образа
```bash
docker build -t taskgen ./taskgen
```

## Генерация задания(режим init)
```bash
docker run --rm -it  taskgen test_task --mode init --seed 1223
```

## Проверка решения студента(режим check)

##### Для задач с текстовым/числовым ответом:

```bash
# Ответ передаётся как строка напрямую в аргументе --solution
docker run --rm taskgen module_3.submodule_5.task_4 --mode check --solution='malloc' --seed 15
```

#####  Для задач с кодом:

```bash
# 1. Создание папки для файлов студента 
mkdir -p /tmp/student_tests

# 2. Запись решения
cat > /tmp/student_tests/solution.c << 'EOF'
// код студента
EOF

# 3. Запуск проверки
docker run --rm -v /tmp/student_tests:/workspace taskgen module_3.submodule_2.task_4 --mode check --solution /workspace/solution.c --seed 15
```
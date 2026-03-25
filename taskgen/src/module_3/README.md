# Модуль №3
Описание задач по ниже перечисленным темам и реализация  module_3.submodule_1.task_1, module_3.submodule_2.task_1, module_3.submodule_2.task_5(инструкция по запуску представлена ниже)

Темы, по которым представлено описание задач:
**указатели**
объявление
инициализация
null
разыменование
адрес переменной
арифметика указателей!
сравнение указателей
указатель на указатель

**массивы**
объявление
инициализация
размер
массив и указатель – в чем разница
передача массива в функцию
многомерные массивы (2-3-4)

**строки**
нультерминаторные строки
string.h
опасность переполнения

**динамическая память**
malloc
calloc
realloc
free

### Инструкция по запуску
### Запуск в режиме init(инициализация задания)

Команда для запуска из папки taskgen:
```bash
python main.py module_3.submodule_1.task_1 --seed 10 --mode init 
```

Пример вывода:
(seed=10)
В каком сегменте памяти хранится локальная переменная внутри функции?

(seed=15)
В каком сегменте памяти хранится динамически выделенная память?

### Запуск в режиме check(проверка решения)

Задача не требующая написания кода:
Команда для запуска из папки taskgen:
```bash
python main.py module_3.submodule_1.task_1  --seed 10 --mode check --solution="Heap"        
```

Пример вывода:
(seed=10, solution="Heap")
Passed: False
FAIL: Ожидалось Stack, получено Heap

(seed=10, solution="Stack")
Passed: True
OK: Верный ответ.

Задача требующая написания кода:
Команда для запуска из папки taskgen:
```bash
python main.py module_3.submodule_2.task_5  --seed 10 --mode check --solution path/to/solution.c        
```

Пример вывода:
(seed=10, solution solution.c)
Passed: True
OK

**Содержимое файла solution.c**
```c
#include <stdio.h>

void print_value(int *ptr) {
    if (ptr == NULL){
        printf("Value: 0\n");
    }
    else
    {
    printf("Value: %d\n", *ptr);
    }
}
``


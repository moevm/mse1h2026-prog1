### Запуск в режиме init(итициализация задания)
Команда для запуска из папки taskgen:
python main.py module_3.submodule_1.task_1 --seed 10 --mode init 
Пример вывода:
(seed=10)
В каком сегменте памяти хранится локальная переменная внутри функции?

(seed=15)
В каком сегменте памяти хранится динамически выделенная память?

### Запуск в режиме check(проверка решения)
Команда для запуска из папки taskgen:
python main.py module_3.submodule_1.task_1  --seed 10 --mode check --solution="Heap"        

Пример вывода:
(seed=10, solution="Heap")
Passed: False
FAIL: Ожидалось Stack, получено Heap

(seed=10, solution="Stack")
Passed: True
OK: Верный ответ.


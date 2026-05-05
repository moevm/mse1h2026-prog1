### Тема: Синтаксис указателя на функцию
**Сложность:** легкая

**Задание:**
Объявите указатель на функцию с именем `{ptr_name}`, которая принимает параметры типов `{param_types}` и возвращает значение типа `{return_type}`.
В ответе напишите только строку объявления, завершающуюся `;`. Имена параметров указывать не обязательно.

**Формат вывода:** `тип (*имя)(типы_параметров);`

**Уникальными значениями становятся:** `ptr_name`, `return_type`, `param_types`
`seed % 4 == 0`: ptr_name = "fp", return_type = "int", param_types = "int, int"
`seed % 4 == 1`: ptr_name = "calc", return_type = "float", param_types = "double, char"
`seed % 4 == 2`: ptr_name = "handler", return_type = "void", param_types = "const char*, int"
`seed % 4 == 3`: ptr_name = "transform", return_type = "char*", param_types = "int, float"

**Пример:** (для seed=15 генерируются `ptr_name` = "transform", `return_type` = "char*", `param_types` = "int, float")
**Ожидаемый вывод:** `char* (*transform)(int, float);`
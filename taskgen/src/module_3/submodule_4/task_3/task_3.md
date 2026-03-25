
#  Тема: "Опасность переполнения"
**Сложность:** средняя
**Задание:**
В коде ниже есть ошибка. Найдите и исправьте её:

```c
include <cstring>

void process_string(const char* source)
{
    char buffer[10];
    strcpy(buffer, source);
}

int main()
{
    const char* input = "This is a long string that exceeds the buffer size.";
    process_string(input);
    return 0;
} 
```
 - Ввод: пустой
 Ввод: пустой
 Пример вывода: snprintf(buffer, sizeof(buffer), "%s", source);




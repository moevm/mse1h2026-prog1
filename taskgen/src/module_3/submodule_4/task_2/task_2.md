# Тема: Библиотека string.h

**Сложность:** средняя

**Задание:** Какая функция соответствует описанию: {description}
Уникальными значениями становятся: description, func_name.
seed % 15 == 0: description = "копирует строку ct в строку s, включая '\0'; возвращает s", func_name = "strcpy"
seed % 15 == 1: description = "копирует не более n символов строки ct в s; возвращает s. Дополняет результат символами '\0', если символов в ct меньше n", func_name = "strncpy"
seed % 15 == 2: description = "приписывает ct к s; возвращает s", func_name = "strcat"
seed % 15 == 3: description = "приписывает не более n символов ct к s, завершая s символом '\0'; возвращает s", func_name = "strncat"
seed % 15 == 4: description = "сравнивает cs и ct; возвращает <0, если cs<ct, 0, если cs==ct, и >0, если cs>ct", func_name = "strcmp"
seed % 15 == 5: description = "сравнивает не более n символов cs и ct; возвращает <0, если cs<ct, 0, если cs==ct, и >0, если cs>ct", func_name = "strncmp"
seed % 15 == 6: description = "возвращает указатель на первое вхождение c в cs или, если такового не оказалось, NULL", func_name = "strchr"
seed % 15 == 7: description = "возвращает указатель на последнее вхождение c в cs или, если такового не оказалось, NULL", func_name = "strrchr"
seed % 15 == 8: description = "возвращает длину начального сегмента cs, состоящего из символов, входящих в строку ct", func_name = "strspn"
seed % 15 == 9: description = "возвращает длину начального сегмента cs, состоящего из символов, не входящих в строку ct", func_name = "strcspn"
seed % 15 == 10: description = "возвращает указатель в cs на первый символ, который совпал с одним из символов, входящих в ct, или NULL", func_name = "strpbrk"
seed % 15 == 11: description = "возвращает указатель на первое вхождение ct в cs или, если такового не оказалось, NULL", func_name = "strstr"
seed % 15 == 12: description = "возвращает длину cs", func_name = "strlen"
seed % 15 == 13: description = "возвращает указатель на зависящую от реализации строку, соответствующую номеру ошибки n", func_name = "strerror"
seed % 15 == 14: description = "ищет в s лексему, ограниченную символами из ct", func_name = "strtok"
Ввод: пустой

Пример для seed=15:
Ввод: пустой
Ожидаемый вывод: strcpy

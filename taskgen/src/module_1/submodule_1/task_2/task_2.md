# Структура программы в си

### Задание №2
- Cложность: средняя.
- Задание: какие из представленных ниже сигнатур функции `main` являются корректными? Укажите номера правильных вариантов через пробел без других знаков
```
signature_1
signature_2
signature_3
signature_4
signature_5
signature_6
signature_7
signature_8
```

- Уникальными становятся значения `signature_1 - signature_8`. Минимум одна сигнатура является корректной. Порядок сигнатур также определяется seed. Возможные варианты `signature_*`: 
1. int main(void)
2. int main()
3. int main(int argc, char *argv[])
4. int main(int argc)
5. int main(char *argv[])
6. int main(int argv, char *argc[])
7. int main(int argc, char *argv)
8. int main(int)
9. int main(char)
10. int main(int argv, char *argc)
11. int main(int argc, char argv[])
12. int main(int argc, char argv)
13. int main(int argv, char argc[])
13. int main(int argv, char argc)
14. и другие аналоги с `void`, `long`, `char` вместо `int`.

- Ввод: пуст. Пример вывода при:
1. int main(char *argv[])
2. void main(void)
3. int main()
4. int main(int argc, char argv)
5. long main(void)
6. char main(int argc, char *argv[])
7. void main(int argc, char argv[])
8. int main(int)

`Ответ: 3.`

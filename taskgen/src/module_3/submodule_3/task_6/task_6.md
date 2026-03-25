# Тема: "Многомерные массивы"
**Сложность:** сложная
## Варианты задач (определяются по seed)

### seed % 2 == 0 : Задача 3.6.1
**Название:** День года по дате

**Условие:**
Напишите функцию `int day_of_year(int year, int month, int day)`, которая:
1. Использует двумерный массив `daytab[2][13]` для хранения дней в месяцах
2. Определяет, високосный ли год (`leap = year%4 == 0 && year%100 != 0 || year%400 == 0`)
3. Преобразует дату (месяц + день) в день года (1–365/366)

**Формат вывода:**
`Day of year: X`

**Пример решения (day_of_year.c):**
```c
#include <stdio.h>

static char daytab[2][13] = {
    {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

void day_of_year(int year, int month, int day) {
    int i, leap;
    leap = year % 4 == 0 && year % 100 != 0 || year % 400 == 0;
    for (i = 1; i < month; i++) {
        day += daytab[leap][i];
    }
    printf("Day of year: %d\n", day);
}

int main() {
    day_of_year(2024, 3, 1);
    return 0;
}
```

### seed % 2 == 1: Задача 3.6.2

**Условие:**
Напишите функцию `void month_day(int year, int yearday, int *pmonth, int *pday)`, которая:
1. Использует двумерный массив `daytab[2][13]` для хранения дней в месяцах
2.  Определяет, високосный ли год (`leap = year%4 == 0 && year%100 != 0 || year%400 == 0`)
3. Преобразует день года (1–365/366) в дату (месяц + день)
4. Записывает результат через указатели `*pmonth` и `*pday`

**Формат вывода:**
`Month: M, Day: D`

**Пример решения (month_day.c):**
```c
#include <stdio.h>

static char daytab[2][13] = {
    {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

void month_day(int year, int yearday, int *pmonth, int *pday) {
    int i, leap;
    leap = year % 4 == 0 && year % 100 != 0 || year % 400 == 0;
    for (i = 1; yearday > daytab[leap][i]; i++) {
        yearday -= daytab[leap][i];
    }
    *pmonth = i;
    *pday = yearday;
    printf("Month: %d, Day: %d\n", m, d);
}

int main() {
    int m, d;
    month_day(2024, 60, &m, &d);
    return 0;
}
```

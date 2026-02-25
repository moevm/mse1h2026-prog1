from memory_model import *
from pointer import *

VALIDATOR_MAP = {
    "1.1.1": Task1_1_1_SegmentByVar,
    "1.2.1": Task1_2_1_AddressFormat,
    "1.3.1": Task1_3_1_DataVsBSS,
    "1.4.1": Task1_4_1_AutomaticLifetime,
    "1.5.1": Task1_5_1_StaticLifetime,
    "1.6.1": Task1_6_1_LocalScope,
    "1.7.1": Task1_7_1_MemorySize,

    "2.1.1": Task2_1_1_DeclarePointer,          
    "2.2.1": Task2_2_1_InitPointer,        
    "2.3.1": Task2_3_1_NullCheckValidator,      
    "2.4.1": Task2_4_1_PointerBasicsValidator,   
    "2.5.1": Task2_5_1_PrintAddressValidator,    
    "2.5.2": Task2_5_2_DereferenceBeforeAfterValidator,  
    "2.6.1": Task2_6_1_PointerIncDecValidator,  
    "2.6.2": Task2_6_2_PointerArithmeticValidator, 
    "2.7.1": Task2_7_1_PointerComparisonValidator,      
    "2.8.1": Task2_8_1_PointerDifferenceValidator,    
    "2.9.1": Task2_9_1_TriplePointerValidator,        
}

def get_validator(task_id: str, seed: int = 0, **kwargs) -> BaseTaskClass:
    if task_id not in VALIDATOR_MAP:
        raise ValueError(f"Неизвестный ID задачи: {task_id}. Доступные: {list(VALIDATOR_MAP.keys())}")
    
    return VALIDATOR_MAP[task_id](seed=seed, **kwargs)


if __name__ == "__main__":

    print("\nЗадача 1.1.1")
    validator = get_validator("1.1.1", seed=1)
    question = validator.init_task()
    print(f"Задание: {question}")
    validator.load_student_solution(solcode="Stack")
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")

    print("\nЗадача 1.2.1")
    validator = get_validator("1.2.1", seed=99)
    question = validator.init_task()
    print(f"Задание: {question}")
    validator.load_student_solution(solcode="Stack")
    success, message= validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")

    print("\nЗадача 1.3.1")
    validator3 = get_validator("1.3.1", seed=99)
    question3 = validator3.init_task()
    print(f"Задание: {question3}")
    validator3.load_student_solution(solcode="Stack")
    success3, message3 = validator3.check()
    print(f"Результат: {'PASS' if success3 else 'FAIL'} - {message3}")

    print("\nЗадача 1.4.1")
    validator4 = get_validator("1.4.1", seed=99)
    question4 = validator4.init_task()
    print(f"Задание: {question4}")
    validator4.load_student_solution(solcode="вниз")
    success4, message4= validator4.check()
    print(f"Результат: {'PASS' if success4 else 'FAIL'} - {message4}")

    print("\nЗадача 1.5.1")
    validator5 = get_validator("1.5.1", seed=99)
    question5 = validator5.init_task()
    print(f"Задание: {question5}")
    validator5.load_student_solution(solcode="Stack")
    success5, message5 = validator5.check()
    print(f"Результат: {'PASS' if success5 else 'FAIL'} - {message5}")

    print("\nЗадача 1.6.1")
    validator6 = get_validator("1.6.1", seed=50)
    question6 = validator6.init_task()
    print(f"Задание: {question6}")
    validator6.load_student_solution(solcode="функция")
    success6, message6= validator6.check()
    print(f"Результат: {'PASS' if success6 else 'FAIL'} - {message6}")

    print("\nЗадача 1.7.1")
    validator7 = get_validator("1.7.1", seed=99)
    question7 = validator7.init_task()
    print(f"Задание: {question7}")
    validator7.load_student_solution(solcode="Heap")
    success7, message7 = validator7.check()
    print(f"Результат: {'PASS' if success7 else 'FAIL'} - {message7}")


    print("\nЗадача 2.1.1: Объявление указателя")
    validator = get_validator("2.1.1")
    question = validator.generate_task()
    print(f"Задание: {question}")
    validator.load_student_solution(solcode="int *ptr;")
    success, message = validator.check()
    print(f"Решение: int *ptr;")
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.2.1: Инициализация указателя")
    validator = get_validator("2.2.1")
    question = validator.generate_task()
    print(f"Задание: {question}")
    validator.load_student_solution(solcode="int a = 10; int *ptr = &a;")
    success, message = validator.check()
    print(f"Решение: int a = 10; int *ptr = &a;")
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.3.1: Проверка на NULL")
    validator = get_validator("2.3.1", seed=42)
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_3_1 = """
#include <stdio.h>
#include <stddef.h>
void solution(int *ptr) {
    if (ptr == NULL) {
        printf("NULL pointer\\n");
    } else {
        printf("Value: %d\\n", *ptr);
    }
}
"""
    validator.load_student_solution(solcode=solution_2_3_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.4.1: Разыменование указателя")
    validator = get_validator("2.4.1", seed=42)
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_4_1 = """
#include <stdio.h>
void solution(int *ptr, int value) {
    *ptr = value;
}
"""
    validator.load_student_solution(solcode=solution_2_4_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.5.1: Вывод адреса переменной")
    validator = get_validator("2.5.1")
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_5_1 = """
#include <stdio.h>
int main() {
    int a = 512;
    int *ptr = &a;
    printf("Переменная a = %d хранится по адресу %p\\n", a, (void*)ptr);
    return 0;
}
"""
    validator.load_student_solution(solcode=solution_2_5_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.5.2: Изменение значения через указатель")
    validator = get_validator("2.5.2", seed=42)
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_5_2 = """
#include <stdio.h>
void solution(int *ptr, int new_value) {
    printf("Before: %d\\n", *ptr);
    *ptr = new_value;
    printf("After: %d\\n", *ptr);
}
"""
    validator.load_student_solution(solcode=solution_2_5_2)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.6.1: Инкремент и декремент указателя")
    validator = get_validator("2.6.1")
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_6_1 = """
#include <stdio.h>
int main() {
    int n = 10;
    int *ptr = &n;
    printf("address=%p value=%d\\n", (void*)ptr, *ptr);
    ptr++;
    printf("address=%p value=%d\\n", (void*)ptr, *ptr);
    ptr--;
    printf("address=%p value=%d\\n", (void*)ptr, *ptr);
    return 0;
}
"""
    validator.load_student_solution(solcode=solution_2_6_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.6.2: Арифметика указателей с массивом")
    validator = get_validator("2.6.2", array_length=5, seed=42)
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_6_2 = """
#include <stdio.h>
void solution(int *arr, int len) {
    int *ptr = arr;
    for(int i = 0; i < len; i++) {
        *ptr = *ptr + 10;
        ptr++;
    }
}
"""
    validator.load_student_solution(solcode=solution_2_6_2)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else ' FAIL'} - {message}")


    print("\nЗадача 2.7.1: Сравнение указателей")
    validator = get_validator("2.7.1", array_length=5, seed=42)
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_7_1 = """
#include <stdio.h>
int solution(int *arr, int len) {
    int *p1 = arr;
    int *p2 = arr + len - 1;
    if (p1 < p2) {
        return 1;
    }
    return 0;
}
"""
    validator.load_student_solution(solcode=solution_2_7_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")

    print("\nЗадача 2.8.1: Разность указателей")
    validator = get_validator("2.8.1")
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_8_1 = """
#include <stdio.h>
int main() {
    int val = 42;
    int *p1 = &val;
    int **p2 = &p1;
    printf("%ld\\n", (long)(p2 - p1));
    return 0;
}
"""
    validator.load_student_solution(solcode=solution_2_8_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")


    print("\nЗадача 2.9.1: Тройной указатель (***ptr)")
    validator = get_validator("2.9.1", seed=42)
    question = validator.generate_task()
    print(f"Задание: {question}")
    solution_2_9_1 = """
#include <stddef.h>
void solution(int ***ptr) {
    ***ptr = 100;
}
"""
    validator.load_student_solution(solcode=solution_2_9_1)
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")

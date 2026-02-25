from build_stages import *
from preprocessor import *

VALIDATOR_MAP = {
    "1.1.1": Task1_1_1_FlagByStage,
    "1.1.2": Task1_1_2_ExtByStage,
    "1.1.3": Task1_1_3_StageByAction,
    "1.1.4": Task1_1_4_DefaultOutput,
    "1.1.5": Task1_1_5_ReadObjectFile,
    "1.2.1": Task1_2_1_GccCommandByStage,
    "1.2.2": Task1_2_2_FullBuildCommand,
    "1.2.3": Task1_2_3_OutputFileByFlag,
    "1.2.4": Task1_2_4_SeparateCompile,
    "1.2.5": Task1_2_5_ErrorStageChoice,
    "1.3.1": Task1_3_1_LinkObjectFiles,
    "1.3.2": Task1_3_2_ContinueBuild,
    "1.3.3": Task1_3_3_TwoFileBuild,

    "2.1.1": Task2_1_1_IncludeAngle,
    "2.1.2": Task2_1_2_IncludeQuotes,
    "2.1.3": Task2_1_3_IncludeSyntax,
    "2.1.4": Task2_1_4_MacroType,
    "2.1.5": Task2_1_5_DirectiveSemicolon,
    "2.1.6": Task2_1_6_DirectiveMeaning,
    "2.2.1": Task2_2_1_MacroExpansion,
    "2.2.2": Task2_2_2_DefineDirective,
    "2.3.1": Task2_3_1_IfndefDefine,
    "2.3.2": Task2_3_2_IfElifElse,
}


def get_validator(task_id: str, seed: int = 0, **kwargs) -> BaseTaskClass:
    if task_id not in VALIDATOR_MAP:
        raise ValueError(f"Неизвестный ID задачи: {task_id}. Доступные: {list(VALIDATOR_MAP.keys())}")
    
    return VALIDATOR_MAP[task_id](seed=seed, **kwargs)


# Проверка работы валидатора
if __name__ == "__main__":
    print("\nПример: Задача 1.1.1")
    
    validator = get_validator("1.1.1", seed=42)
    question = validator.init_task()
    print(f"Вопрос: {question}")
    validator.load_student_solution(solcode="-E")
    success, message = validator.check()
    print(f"Ответ '-E': {'PASS' if success else 'FAIL'} - {message}")


    print("\nПример: Задача 1.1.2")
    validator2 = get_validator("1.1.2", seed=57)
    question2 = validator2.init_task()
    print(f"Вопрос: {question2}")
    validator2.load_student_solution(solcode="-X")
    success2, message2 = validator2.check()
    print(f"Ответ '-X': {'PASS' if success2 else 'FAIL'} - {message2}")
    

    print("\nПример: Задача 1.2.1")
    validator3 = get_validator("1.2.1", seed=100)
    question3 = validator3.init_task()
    print(f"Вопрос: {question3}")
    validator3.load_student_solution(solcode="gcc -S prog100.c -o prog100.s")
    success3, message3 = validator3.check()
    print(f"Результат: {'PASS' if success3 else 'FAIL'} - {message3}")


    print("\nПример: Задача 2.1.1")
    validator = get_validator("2.1.1", seed=42)
    question = validator.init_task()
    print(f"Вопрос: {question}")
    validator.load_student_solution(solcode="в системных директориях")
    success, message = validator.check()
    print(f"Результат: {'PASS' if success else 'FAIL'} - {message}")
    

    print("\nПример: Задача 2.2.2")
    validator2 = get_validator("2.2.2", seed=100)
    question2 = validator2.init_task()
    print(f"Вопрос: {question2}")
    validator2.load_student_solution(solcode="#define MACRO100(x) ((x) * 2)")
    success2, message2 = validator2.check()
    print(f"Результат: {'PASS' if success2 else 'FAIL'} - {message2}")
    

    print("\nПример: Задача 2.3.1")
    validator3 = get_validator("2.3.1", seed=32)
    question3 = validator3.init_task()
    print(f"Вопрос: {question3}")
    validator3.load_student_solution(solcode="#ifndef MACRO50\n#define MACRO50 75\n#endif")
    success3, message3 = validator3.check()
    print(f"Результат: {'PASS' if success3 else 'FAIL'} - {message3}")

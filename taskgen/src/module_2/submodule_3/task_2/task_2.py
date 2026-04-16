from src.base_module.base_task import BaseTaskClass
import random

class Module2HeaderFileTask2(BaseTaskClass):
    """Задание №3.2.1"""

    _TRUE = [
        "не является частью стандарта C",
        "зависит от реализации компилятора",
        "защищает файл от повторного включения",
        "имеет более короткий синтаксис чем #ifndef",
        "не требует уникального имени макроса",
        "поддерживается большинством современных компиляторов",
        "может ускорить сборку проекта",
        "избегает конфликтов имен макросов-гардов"
    ]
    _FALSE = [
        "является частью стандарта C99",
        "гарантированно работает на всех компиляторах",
        "требует указания уникального имени макроса",
        "работает медленнее чем #ifndef",
        "не защищает от повторного включения",
        "обязательна к использованию в языке C",
        "позволяет включать файл частично при повторном чтении"
    ]


    def __init__(self, seed: int):
        super().__init__(seed)
        self._seed_local = int(seed)
        self._statements, self._correct_indices, self._expected_answer = self._generate_params()
        self.student_solution = ""
    
    def _generate_params(self):
        rng = random.Random(self._seed_local)

        num_true = rng.randint(1, 3)
        true_selected = rng.sample(self._TRUE, num_true)
        false_selected = rng.sample(self._FALSE, 4 - num_true)

        combined = [(s, True) for s in true_selected] + [(s, False) for s in false_selected]
        rng.shuffle(combined)
        
        statements = [s[0] for s in combined]
        
        correct_indices = sorted([i + 1 for i, (_, is_true) in enumerate(combined) if is_true])
        expected_answer = " ".join(map(str, correct_indices))
        
        return statements, correct_indices, expected_answer

    def generate_task(self):
        opts = "\n".join([f"{i+1}. {s}" for i, s in enumerate(self._statements)])
        self.task_text = (
            f"Выберите верные утверждения о директиве #pragma once:\n{opts}\nВ ответе укажите номера верных утверждений через пробел или запятую."
        )

    def init_task(self):
        self.generate_task()
        return self.task_text

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self) -> tuple[bool, str]:
        student_answer = getattr(self, "student_solution", "").strip()
        try:
            student_indices = set(
                int(x) for x in student_answer.replace(',', ' ').replace('\n', ' ').split() if x
            )
        except ValueError:
            return False, "FAIL: Ожидаются номера вариантов (например: '1 3')."

        correct_set = set(self._correct_indices)

        if student_indices == correct_set:
            return True, "OK: Верный ответ."
        elif not student_indices.issubset(correct_set):
            return False, f"FAIL: Неверный ответ."
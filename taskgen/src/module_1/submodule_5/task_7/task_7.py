from typing import Optional, List, Tuple
import random
from src.base_module.base_task import BaseTaskClass, TestItem


class ExprNode:
    __slots__ = ('type', 'value', 'op', 'left', 'right')
    def __init__(self, type_: str, value: int = None, op: str = None,
                 left: 'ExprNode' = None, right: 'ExprNode' = None):
        self.type = type_
        self.value = value
        self.op = op
        self.left = left
        self.right = right


def generate_ast(depth: int, rng: random.Random) -> ExprNode:
    if depth >= 3 or rng.random() < 0.3:
        val = rng.randint(-10, 10)
        return ExprNode('int', value=val)
    if rng.random() < 0.4:
        op = rng.choice(['&&', '||'])
        left = generate_ast(depth+1, rng)
        right = generate_ast(depth+1, rng)
        return ExprNode('binary', op=op, left=left, right=right)
    if rng.random() < 0.5:
        left = generate_ast(depth+1, rng)
        return ExprNode('unary', op='!', left=left)
    op = rng.choice(['+', '-', '*', '/', '%'])
    left = generate_ast(depth+1, rng)
    if op in ('/', '%'):
        right = ExprNode('int', value=rng.randint(1, 10))
    else:
        right = generate_ast(depth+1, rng)
    return ExprNode('binary', op=op, left=left, right=right)


def to_string(node: ExprNode, parent_prec: int = 0, is_right: bool = False) -> str:
    prec = {'||': 1, '&&': 2, '!': 3, '+': 4, '-': 4, '*': 5, '/': 5, '%': 5}
    if node.type == 'int':
        return str(node.value)
    if node.type == 'unary':
        inner = to_string(node.left, prec['!'], False)
        if node.left.type == 'binary' and prec.get(node.left.op, 0) < prec['!']:
            inner = f"({inner})"
        return f"!{inner}"
    self_prec = prec.get(node.op, 0)
    left_str = to_string(node.left, self_prec, False)
    right_str = to_string(node.right, self_prec, True)
    if node.left.type == 'binary' and prec.get(node.left.op, 0) < self_prec:
        left_str = f"({left_str})"
    if node.right.type == 'binary' and prec.get(node.right.op, 0) < self_prec:
        right_str = f"({right_str})"
    elif node.right.type == 'binary' and prec.get(node.right.op, 0) == self_prec and node.op in ('-', '/', '%', '&&', '||'):
        right_str = f"({right_str})"
    return f"{left_str} {node.op} {right_str}"


def evaluate_and_collect(node: ExprNode) -> Tuple[int, List[str]]:
    if node.type == 'int':
        return node.value, []
    if node.type == 'unary':
        val, ops = evaluate_and_collect(node.left)
        return (1 if val == 0 else 0), ops + ['!']
    if node.op == '&&':
        left_val, left_ops = evaluate_and_collect(node.left)
        if left_val == 0:
            return 0, left_ops + ['&&']
        right_val, right_ops = evaluate_and_collect(node.right)
        return (1 if right_val != 0 else 0), left_ops + right_ops + ['&&']
    if node.op == '||':
        left_val, left_ops = evaluate_and_collect(node.left)
        if left_val != 0:
            return 1, left_ops + ['||']
        right_val, right_ops = evaluate_and_collect(node.right)
        return (1 if right_val != 0 else 0), left_ops + right_ops + ['||']
    left_val, left_ops = evaluate_and_collect(node.left)
    right_val, right_ops = evaluate_and_collect(node.right)
    if node.op == '+':
        return left_val + right_val, left_ops + right_ops
    if node.op == '-':
        return left_val - right_val, left_ops + right_ops
    if node.op == '*':
        return left_val * right_val, left_ops + right_ops
    if node.op == '/':
        if right_val == 0:
            return 0, left_ops + right_ops
        return int(left_val / right_val), left_ops + right_ops
    if node.op == '%':
        if right_val == 0:
            return 0, left_ops + right_ops
        res = left_val - (int(left_val / right_val)) * right_val
        return res, left_ops + right_ops
    return 0, []


class Module_1_Submodule_5_task_7(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rng = random.Random(self.seed)
        self.ast = generate_ast(0, rng)
        self.expr_str = to_string(self.ast)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (f"Есть выражение `{self.expr_str}`. Напишите логические операции в порядке их выполнения "
                f"через пробел, затем через пробел укажите значение этого выражения.")

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        _, ops = evaluate_and_collect(self.ast)
        val, _ = evaluate_and_collect(self.ast)
        expected = " ".join(ops) + f" {val}"
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=expected,
                compare_func=lambda out, exp: out.strip() == exp.strip()
            )
        ]

    def run_solution(self, test: TestItem):
        student_answer = self.student_solution.strip()
        if test.compare_func(student_answer, test.expected):
            return None
        return student_answer, test.expected

    def load_student_solution(self, solution):
        if not solution.strip():
            raise ValueError("Решение пустое.")
        self.student_solution = solution.strip()

    def check(self):
        try:
            self.generate_task()
            _, ops = evaluate_and_collect(self.ast)
            val, _ = evaluate_and_collect(self.ast)
            expected = " ".join(ops) + f" {val}"
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
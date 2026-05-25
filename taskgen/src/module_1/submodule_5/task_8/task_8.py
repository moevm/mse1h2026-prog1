from typing import Optional
import random
from src.base_module.base_task import BaseTaskClass, TestItem


def to_int32(x: int) -> int:
    x &= 0xFFFFFFFF
    if x & 0x80000000:
        x -= 0x100000000
    return x


class ExprNode:
    __slots__ = ('type', 'value', 'op', 'left', 'right')
    def __init__(self, type_, value=None, op=None, left=None, right=None):
        self.type = type_
        self.value = value
        self.op = op
        self.left = left
        self.right = right


def generate_ast(depth: int, rng: random.Random) -> ExprNode:
    if depth >= 2 or rng.random() < 0.3:
        val = rng.randint(-50, 50)
        return ExprNode('int', value=val)
    if rng.random() < 0.2:
        left = generate_ast(depth+1, rng)
        return ExprNode('unary', op='~', left=left)
    ops = ['&', '|', '^', '<<', '>>']
    op = rng.choice(ops)
    left = generate_ast(depth+1, rng)
    right = generate_ast(depth+1, rng)
    return ExprNode('binary', op=op, left=left, right=right)


def to_string(node: ExprNode, parent_prec: int = 0, is_right: bool = False) -> str:
    prec = {'~': 3, '<<': 2, '>>': 2, '&': 1, '^': 0, '|': -1}
    if node.type == 'int':
        return str(node.value)
    if node.type == 'unary':
        inner = to_string(node.left, prec['~'], False)
        if node.left.type == 'binary' and prec.get(node.left.op, -2) < prec['~']:
            inner = f"({inner})"
        return f"~{inner}"
    self_prec = prec[node.op]
    left_str = to_string(node.left, self_prec, False)
    right_str = to_string(node.right, self_prec, True)
    if node.left.type == 'binary' and prec.get(node.left.op, -2) < self_prec:
        left_str = f"({left_str})"
    if node.right.type == 'binary' and prec.get(node.right.op, -2) < self_prec:
        right_str = f"({right_str})"
    elif node.right.type == 'binary' and prec.get(node.right.op, -2) == self_prec and node.op in ('<<', '>>', '-', '/', '%'):
        right_str = f"({right_str})"
    return f"{left_str} {node.op} {right_str}"


def evaluate(node: ExprNode) -> int:
    if node.type == 'int':
        return to_int32(node.value)
    if node.type == 'unary':
        val = evaluate(node.left)
        return to_int32(~val)
    left = evaluate(node.left)
    right = evaluate(node.right)
    if node.op == '&':
        return to_int32(left & right)
    if node.op == '|':
        return to_int32(left | right)
    if node.op == '^':
        return to_int32(left ^ right)
    if node.op == '<<':
        return to_int32(left << right)
    if node.op == '>>':
        return to_int32(left >> right)
    return 0


class Module_1_Submodule_5_task_8(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rng = random.Random(self.seed)
        self.ast = generate_ast(0, rng)
        self.expr_str = to_string(self.ast)
        self.result = evaluate(self.ast)
        self.student_solution = ""

    def generate_task(self) -> str:
        return f"Есть выражение `{self.expr_str}`. Найдите значение этого выражения."

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        self.tests = [
            TestItem(
                input_str="",
                showed_input="",
                expected=str(self.result),
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
            student = self.student_solution.strip()
            if student == str(self.result):
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
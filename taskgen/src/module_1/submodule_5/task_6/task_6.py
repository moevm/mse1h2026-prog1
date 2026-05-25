from typing import Optional, Union
import random
from src.base_module.base_task import BaseTaskClass, TestItem


class ExprNode:
    __slots__ = ('type', 'value', 'op', 'left', 'right')
    def __init__(self, type_: str, value: Union[int, None] = None, op: str = None,
                 left: 'ExprNode' = None, right: 'ExprNode' = None):
        self.type = type_
        self.value = value
        self.op = op
        self.left = left
        self.right = right


TOTAL_OPS = 6


def generate_ast_fixed(remaining_ops: int, rng: random.Random) -> ExprNode:
    if remaining_ops == 0:
        value = rng.randint(-50, 50)
        return ExprNode('int', value=value)

    if rng.random() < 0.8:
        ops = ['+', '-', '*', '/', '%', '&&', '||']
        op = rng.choice(ops)
        total_child_ops = remaining_ops - 1
        left_ops = rng.randint(0, total_child_ops)
        right_ops = total_child_ops - left_ops
        left = generate_ast_fixed(left_ops, rng)
        right = generate_ast_fixed(right_ops, rng)

        if op in ('/', '%') and right.type == 'int' and right.value == 0:
            right.value = 1

        return ExprNode('binary', op=op, left=left, right=right)
    else:
        child_ops = remaining_ops - 1
        child = generate_ast_fixed(child_ops, rng)
        return ExprNode('unary', op='!', left=child)


def to_string(node: ExprNode, parent_op: str = None, is_right: bool = False) -> str:
    if node.type == 'int':
        return str(node.value)
    if node.type == 'unary':
        inner = to_string(node.left, '!', False)
        return f"!{inner}" if inner.startswith('(') or node.left.type == 'int' else f"!({inner})"
    prec = {'||': 1, '&&': 2, '+': 3, '-': 3, '*': 4, '/': 4, '%': 4}
    self_prec = prec.get(node.op, 0)
    left_str = to_string(node.left, node.op, False)
    right_str = to_string(node.right, node.op, True)
    left_need = False
    if node.left.type == 'binary':
        left_prec = prec.get(node.left.op, 0)
        if left_prec < self_prec:
            left_need = True
        elif left_prec == self_prec and node.op in ('-', '/', '%'):
            pass
    if left_need:
        left_str = f"({left_str})"
    right_need = False
    if node.right.type == 'binary':
        right_prec = prec.get(node.right.op, 0)
        if right_prec < self_prec:
            right_need = True
        elif right_prec == self_prec and node.op in ('-', '/', '%', '&&', '||'):
            right_need = True
    if right_need:
        right_str = f"({right_str})"
    return f"{left_str} {node.op} {right_str}"


def evaluate(node: ExprNode) -> int:
    if node.type == 'int':
        return node.value
    if node.type == 'unary':
        left_val = evaluate(node.left)
        return 1 if not left_val else 0
    left = evaluate(node.left)
    right = evaluate(node.right)
    op = node.op
    if op == '+':
        return left + right
    if op == '-':
        return left - right
    if op == '*':
        return left * right
    if op == '/':
        if right == 0:
            return 0
        return int(left / right)
    if op == '%':
        if right == 0:
            return 0
        return left % right
    if op == '&&':
        return 1 if (left != 0 and right != 0) else 0
    if op == '||':
        return 1 if (left != 0 or right != 0) else 0
    return 0


def collect_order(node: ExprNode) -> str:
    if node.type == 'int':
        return ''
    if node.type == 'unary':
        left_order = collect_order(node.left)
        return left_order + node.op
    left_order = collect_order(node.left)
    right_order = collect_order(node.right)
    return left_order + right_order + node.op


class Module_1_Submodule_5_task_6(BaseTaskClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rng = random.Random(self.seed)
        self.ast = generate_ast_fixed(TOTAL_OPS, rng)
        self.expression_str = to_string(self.ast)
        self.value = evaluate(self.ast)
        self.order_str = collect_order(self.ast)
        self.student_solution = ""

    def generate_task(self) -> str:
        return (f"Есть выражение `{self.expression_str}`. Напишите операции в порядке их выполнения "
                f"(без пробела, скобки не считаются), затем через пробел укажите значение этого выражения.")

    def compile(self) -> Optional[str]:
        return None

    def _generate_tests(self):
        expected = f"{self.order_str} {self.value}"
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
            expected = f"{self.order_str} {self.value}"
            student = self.student_solution.strip()
            if student == expected:
                return True, "OK: Верный ответ."
            else:
                return False, "FAIL: Ответ неверный."
        except Exception as e:
            return False, f"FAIL: {str(e)}"
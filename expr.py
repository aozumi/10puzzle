from typing import List, Callable, Dict, Generator, TypeVar, Sequence
from functools import reduce

### ===
### 式
### ===

# 演算子
# -----

BinaryOperator = Callable[[float, float], float]

def op_add(x: float, y: float) -> float: return x + y
def op_sub(x: float, y: float) -> float: return x - y
def op_mul(x: float, y: float) -> float: return x * y
def op_div(x: float, y: float) -> float: return x / y

OPERATORS = [op_add, op_sub, op_mul, op_div]

# OPERATORSのうち、交換法則が成り立つもの
CUMMULATIVE_OPERATORS = [op_add, op_mul]

# 演算子の文字表現
OPNAMES: Dict[BinaryOperator, str] = {
    op_add: '+',
    op_sub: '-',
    op_mul: '*',
    op_div: '/'
}

# 式
# ---

class Expr:
    def eval(self) -> float:
        raise NotImplemented

    def is_equiv(self, other) -> bool:
        """
        2つの式が(交換法則を考慮して)同じ式と言えるなら真を返す。
        """
        return self == other

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self}>"


class BinaryOp (Expr):
    """
    2項演算ノード。
    """

    def __init__(self, op: BinaryOperator, left: Expr, right: Expr) -> None:
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def eval(self) -> float:
        l = self.left.eval()
        r = self.right.eval()
        return self.op(l, r)

    def __eq__(self, other) -> bool:
        if other and isinstance(other, BinaryOp):
            return (self.op == other.op
                    and self.left == other.left
                    and self.right == other.right)
        else:
            return False

    def is_equiv(self, other) -> bool:
        if not isinstance(other, BinaryOp):
            return False
        if self.op != other.op:
            return False
        if (self.left.is_equiv(other.left) and
            self.right.is_equiv(other.right)):
            return True
        if self.op in CUMMULATIVE_OPERATORS:
            return (self.left.is_equiv(other.right) and
                    self.right.is_equiv(other.left))
        return False

    def operand_str(self, node: Expr) -> str:
        if isinstance(node, Value):
            return str(node)
        else:
            return f"({node})"

    def __str__(self) -> str:
        left_s = self.operand_str(self.left)
        right_s = self.operand_str(self.right)
        return f"{left_s}{OPNAMES[self.op]}{right_s}"


class Value (Expr):
    """
    定数項ノード。
    """
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def eval(self) -> float:
        return self.value

    def __eq__(self, other) -> bool:
        if other and isinstance(other, Value):
            return self.value == other.value
        else:
            return False

    def __str__(self) -> str:
        return str(self.value)


def build_exprs(nums: Sequence[float], ops: Sequence[BinaryOperator]) -> Generator[Expr, None, None]:
    """
    項 nums, 二項演算子 ops からなる式を全て生成する。
    """

    assert len(nums) == len(ops) + 1

    if not ops:
        yield Value(nums[0])
        return

    # どの2項演算をルートノードにするかを変えながらループする
    for op_index in range(len(ops)):
        left_exprs = build_exprs(nums[:op_index+1], ops[:op_index])
        op = ops[op_index]
        right_exprs = list(build_exprs(nums[op_index+1:], ops[op_index+1:]))
        for l in left_exprs:
            for r in right_exprs:
                yield BinaryOp(op, l, r)


def remove_duplicated_exprs(exprs: Sequence[Expr]) -> Sequence[Expr]:
    """
    exprs から重複する式を除いたリストを返す。
    """
    exprs2 = []
    for expr in exprs:
        if any(expr.is_equiv(e) for e in exprs2):
            continue
        exprs2.append(expr)

    return exprs2

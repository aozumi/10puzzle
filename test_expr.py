from expr import *


class TestValue:
    def test_str(self):
        node = Value(3)
        assert str(node) == "3"

    def test_eval(self):
        node = Value(10)
        assert node.eval() == 10

    def test_eq(self):
        node1 = Value(3)
        node2 = Value(3)
        assert node1 == node2

    def test_eq__not_equal(self):
        node1 = Value(3)
        node2 = Value(4)
        assert not (node1 == node2)

    def test_is_equiv(self):
        node1 = Value(3)
        node2 = Value(3)
        assert node1.is_equiv(node2)

    def test_is_equiv__not_equiv(self):
        node1 = Value(3)
        node2 = Value(4)
        assert not node1.is_equiv(node2)


class TestBinaryOp:
    def test_str(self):
        node = BinaryOp(op_add, Value(3), Value(5))
        assert str(node) == "3+5"

    def test_str__left_binop(self):
        node = BinaryOp(op_add,
                        BinaryOp(op_sub, Value(7), Value(4)),
                        Value(5))
        assert str(node) == "(7-4)+5"

    def test_str__right_binop(self):
        node = BinaryOp(op_add,
                        Value(5),
                        BinaryOp(op_sub, Value(7), Value(4)))
        assert str(node) == "5+(7-4)"

    def test_eval(self):
        node = BinaryOp(op_sub, Value(10), Value(7))
        assert node.eval() == 3

    def test_eq(self):
        node1 = BinaryOp(op_add, Value(3), Value(4))
        node2 = BinaryOp(op_add, Value(3), Value(4))
        assert node1 == node2

    def test_eq__op_is_different(self):
        node1 = BinaryOp(op_add, Value(3), Value(4))
        node2 = BinaryOp(op_sub, Value(3), Value(4))
        assert not (node1 == node2)

    def test_eq__left_is_different(self):
        node1 = BinaryOp(op_add, Value(3), Value(4))
        node2 = BinaryOp(op_add, Value(4), Value(4))
        assert not (node1 == node2)

    def test_eq__right_is_different(self):
        node1 = BinaryOp(op_add, Value(3), Value(4))
        node2 = BinaryOp(op_add, Value(3), Value(5))
        assert not (node1 == node2)

    def test_is_equiv__simple_add(self):
        node1 = BinaryOp(op_add, Value(3), Value(4))
        node2 = BinaryOp(op_add, Value(4), Value(3))
        assert node1.is_equiv(node2)

    def test_is_equiv__simple_mul(self):
        node1 = BinaryOp(op_mul, Value(3), Value(4))
        node2 = BinaryOp(op_mul, Value(4), Value(3))
        assert node1.is_equiv(node2)

    def test_is_equiv__simple_sub(self):
        node1 = BinaryOp(op_sub, Value(3), Value(4))
        node2 = BinaryOp(op_sub, Value(4), Value(3))
        assert not node1.is_equiv(node2)

    def test_is_equiv__simple_div(self):
        node1 = BinaryOp(op_div, Value(3), Value(4))
        node2 = BinaryOp(op_div, Value(4), Value(3))
        assert not node1.is_equiv(node2)


def test_build_exprs__single_binop() -> None:
    NUMS = [3, 5]
    OPS = [op_add]
    actual = list(build_exprs(NUMS, OPS))
    assert len(actual) == 1
    assert str(actual[0]) == "3+5"
    assert actual[0] == BinaryOp(op_add, Value(3), Value(5))


def test_build_exprs__single_atom() -> None:
    actual = list(build_exprs([3], []))
    assert str(actual[0]) == "3"
    assert actual == [Value(3)]


def test_build_exprs__2ops():
    NUMS = [3, 5, 7]
    OPS = [op_add, op_sub]
    actual = list(build_exprs(NUMS, OPS))
    expected = [
        BinaryOp(op_add,
                 Value(3),
                 BinaryOp(op_sub, Value(5), Value(7))),
        BinaryOp(op_sub,
                 BinaryOp(op_add, Value(3), Value(5)),
                 Value(7)),
    ]
    assert list(map(str, actual)) == ["3+(5-7)", "(3+5)-7"]
    assert actual == expected

def test_build_exprs__3ops():
    NUMS = [3, 5, 7, 9]
    OPS = [op_add, op_sub, op_mul]
    actual = list(build_exprs(NUMS, OPS))
    expected = [
        # 3+(5-(7*9)),
        BinaryOp(op_add,
                 Value(3),
                 BinaryOp(op_sub,
                          Value(5),
                          BinaryOp(op_mul,
                                   Value(7),
                                   Value(9)))),
        # 3+((5-7)*9),
        BinaryOp(op_add,
                 Value(3),
                 BinaryOp(op_mul,
                          BinaryOp(op_sub,
                                   Value(5),
                                   Value(7)),
                          Value(9))),
        # (3+5)-(7*9),
        BinaryOp(op_sub,
                 BinaryOp(op_add,
                          Value(3),
                          Value(5)),
                 BinaryOp(op_mul,
                          Value(7),
                          Value(9))),
        # (3+(5-7))*9,
        BinaryOp(op_mul,
                 BinaryOp(op_add,
                          Value(3),
                          BinaryOp(op_sub,
                                   Value(5),
                                   Value(7))),
                 Value(9)),
        # ((3+5)-7)*9,
        BinaryOp(op_mul,
                 BinaryOp(op_sub,
                          BinaryOp(op_add,
                                   Value(3),
                                   Value(5)),
                          Value(7)),
                 Value(9)),
    ]
    assert actual == expected

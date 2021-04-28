from simpleexpr import SimpleExpr, Value, AddSub, MulDiv
from simpleexpr import build_exprs, signs_to_str, single_exprs


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


def wrapvalue(args):
    for x in args:
        if isinstance(x, float) or isinstance(x, int):
            yield Value(x)
        else:
            yield x


def addsub(addargs, subargs):
    return AddSub(wrapvalue(addargs), wrapvalue(subargs))


def muldiv(mulargs, divisors):
    return MulDiv(wrapvalue(mulargs), wrapvalue(divisors))


def add(*args):
    return addsub(args, [])


def sub(*args):
    return addsub(args[:1], args[1:])


def mul(*args):
    return muldiv(args, [])


def div(*args):
    return muldiv(args[:1], args[1:])


class TestAddSub:
    def test_str(self):
        node = AddSub([Value(3), Value(4)], [Value(5), Value(7)])
        assert str(node) == "3+4-5-7"

    def test_str__has_muldiv(self):
        node = AddSub([Value(3), MulDiv([Value(4), Value(6)], [])],
                      [Value(5), MulDiv([Value(7), Value(8)], [])])
        assert str(node) == "3+4*6-5-7*8"

    def test_eval(self):
        node = AddSub([Value(1), Value(2)], [Value(5), Value(9)])
        assert node.eval() == -11

    def test_eq(self):
        node1 = AddSub([Value(3)], [Value(4)])
        node2 = AddSub([Value(3)], [Value(4)])
        assert node1 == node2

    def test_eq__noteq(self):
        node1 = AddSub([Value(3)], [Value(4)])
        node2 = AddSub([Value(4)], [Value(4)])
        assert not (node1 == node2)

    def test_eq__muldiv(self):
        node1 = addsub([muldiv([3], [4])], [])
        node2 = addsub([muldiv([3], [4])], [])
        assert node1 == node2


class TestMulDiv:
    def test_str(self):
        node = MulDiv([Value(3), Value(4)], [Value(5), Value(7)])
        assert str(node) == "3*4/5/7"

    def test_str__has_addsub(self):
        node = MulDiv([Value(3), AddSub([Value(4), Value(6)], [])],
                      [Value(5), AddSub([Value(7), Value(8)], [])])
        assert str(node) == "3*(4+6)/5/(7+8)"

    def test_eval(self):
        node = MulDiv([Value(6), Value(10)], [Value(5), Value(3)])
        assert node.eval() == 4

    def test_eq(self):
        node1 = MulDiv([Value(3)], [Value(4)])
        node2 = MulDiv([Value(3)], [Value(4)])
        assert node1 == node2

    def test_eq__noteq(self):
        node1 = MulDiv([Value(3)], [Value(4)])
        node2 = MulDiv([Value(4)], [Value(4)])
        assert not (node1 == node2)

    def test_eq__addsub(self):
        node1 = muldiv([addsub([3], [4])], [])
        node2 = muldiv([addsub([3], [4])], [])
        assert node1 == node2


def test_signs_to_str():
    assert signs_to_str([1, -1, 1, 1]) == '+-++'


def test_build_exprs__single_binop() -> None:
    NUMS = [3, 5]
    actual = list(build_exprs(NUMS))
    expected = [
        add(3, 5),
        sub(3, 5),
        sub(5, 3),
        mul(3, 5),
        div(3, 5),
        div(5, 3),
    ]
    assert actual == expected


def test_build_exprs__single_atom() -> None:
    actual = list(build_exprs([3]))
    assert actual == [Value(3)]


def binary_expr(op1, op2, a, b):
    return [
        op1(a, b),
        op2(a, b),
        op2(b, a)
    ]

def binary_addsub(a, b): return binary_expr(add, sub, a, b)
def binary_muldiv(a, b): return binary_expr(mul, div, a, b)

def ternary_expr(op, a, b, c):
    return [
        op([a, b, c], []),
        op([a, b], [c]),
        op([a, c], [b]),
        op([a], [b, c]),
        op([b, c], [a]),
        op([b], [a, c]),
        op([c], [a, b]),
    ]

def ternary_addsub(a, b, c): return ternary_expr(addsub, a, b, c)
def ternary_muldiv(a, b, c): return ternary_expr(muldiv, a, b, c)

def flatten(xs):
    for x in xs:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x

def test_build_exprs__2ops():
    NUMS = [3, 5, 7]

    exprs_35_7 = ([binary_muldiv(7, e) for e in binary_addsub(3, 5)] +
                  [binary_addsub(7, e) for e in binary_muldiv(3, 5)])
    exprs_37_5 = ([binary_muldiv(5, e) for e in binary_addsub(3, 7)] +
                  [binary_addsub(5, e) for e in binary_muldiv(3, 7)])
    exprs_57_3 = ([binary_muldiv(3, e) for e in binary_addsub(5, 7)] +
                  [binary_addsub(3, e) for e in binary_muldiv(5, 7)])
    exprs_357 = ternary_addsub(3, 5, 7) + ternary_muldiv(3, 5, 7)
    expected = list(flatten(exprs_35_7 + exprs_37_5 + exprs_57_3 + exprs_357))
    actual = list(build_exprs(NUMS))
    assert expected == actual


def test_single_exprs__1_2():
    terms = list(map(Value, [1, 2]))
    assert set(single_exprs(terms)) == set([
        add(1, 2),
        sub(1, 2),
        sub(2, 1),
        mul(1, 2),
        div(1, 2),
    ])

def test_single_exprs__1_2_3():
    terms = list(map(Value, [1, 2, 3]))
    assert set(single_exprs(terms)) == set([
        addsub([1, 2, 3], []),
        addsub([1, 2], [3]),
        addsub([1, 3], [2]),
        addsub([1], [2, 3]),
        addsub([2, 3], [1]),
        addsub([2], [1, 3]),
        # addsub([3], [1, 2]), 0 になる AddSub の重複排除により生成されない
        muldiv([1, 2, 3], []),
        muldiv([1, 2], [3]),
        # muldiv([2, 3], [1]),
        muldiv([1, 3], [2]),
        muldiv([1], [2, 3]),
        # muldiv([2], [1, 3]),
        # muldiv([3], [1, 2]),
    ])

def test_single_exprs__2_3_4():
    terms = list(map(Value, [2, 3, 4]))
    assert set(single_exprs(terms)) == set([
        addsub([2, 3, 4], []),
        addsub([2, 3], [4]),
        addsub([2, 4], [3]),
        addsub([2], [3, 4]),
        addsub([3, 4], [2]),
        addsub([3], [2, 4]),
        addsub([4], [2, 3]),
        muldiv([2, 3, 4], []),
        muldiv([2, 3], [4]),
        muldiv([2, 4], [3]),
        muldiv([2], [3, 4]),
        muldiv([3, 4], [2]),
        muldiv([3], [2, 4]),
        muldiv([4], [2, 3]),
    ])

def test_single_exprs__1sub1_2_3():
    terms = [Value(2), Value(3), sub(1, 1)]
    # 0となる項を含むので乗算のみ生成
    assert set(single_exprs(terms)) == set([
        MulDiv([Value(2), Value(3), sub(1, 1)], [])
    ])


# def test_operators__empty():
#     assert operators(2, False, False) == []


# def test_operators__2a():
#     assert [x.name for x in operators(2, True, False)] == [
#         'addsub++',
#         'addsub+-',
#         'addsub-+',
#     ]


# def test_operators__2m():
#     assert [x.name for x in operators(2, False, True)] == [
#         'muldiv++',
#         'muldiv+-',
#         'muldiv-+',
#     ]


# def test_operators__2am():
#     assert [x.name for x in operators(2, True, True)] == [
#         'addsub++',
#         'addsub+-',
#         'addsub-+',
#         'muldiv++',
#         'muldiv+-',
#         'muldiv-+',
#     ]

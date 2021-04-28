from __future__ import annotations
from functools import reduce
from typing import Union, List, Literal, Sequence, Generator, Iterable, TypeVar, cast, Optional, Callable, Tuple, Dict, Set
from itertools import combinations, product
import logging


LOG = logging.getLogger(__name__)


def add(args: Iterable[float]) -> float:
    return reduce(lambda x, y: x + y, args, cast(float, 0))


def mul(args: Iterable[float]) -> float:
    return reduce(lambda x, y: x * y, args, cast(float, 1))

# 式の表現
# --------
# Value, AddSub, MulDiv の3種類のノードで構文木を構成する。
#
# 四則演算をナイーブに2項演算として二分木で表すと、
# 等価な式がいくつもの木構造で表現できてしまうので
# 重複する式の排除や探索効率の面で不便である。
#
# そこで、構文木の表現を次のようにする。
#
# - 加算と減算を表現する AddSub ノード: 任意個の項の加算・減算をまとめて表す。
#   オペランドごとに演算の種類(加算か減算か)の情報を持つ。
#   オペランドの並び順は持たない。
# - 乗算と除算を表現する MulDiv ノード: 任意個の項の乗算・除算をまとめて表す
#   オペランドごとに演算の種類(乗算か除算か)の情報を持つ。
#   オペランドの並び順は持たない。
# - AddSub、MulDiv はそれぞれ自分と同じクラスのノードをオペランドにとらない。
#
# 式の中の項の並びは持たず、ノードを生成するときは項の中でどの項をオペランドと
# するかだけを選択する。項の並び順は構文木から式を出力する際に決める。
#
# AddSub ノードを生成するときは、選択された項の全ての加算・減算の組み合わせを
# 同時に生成する。例えば A, B, C という3項の AddSub ノードを生成する場合、
# A+B+C, A+B-C, A-B+C, A-B-C, -A+B+C, -A+B-C, -A-B+C の7通りの組み合わせを
# 生成する(減算には被減数が必要なので、-A-B-C は生成しない)。
# MulDiv ノードの場合も同様にする。
#
# これによって、等価な式の重複生成をかなり抑制できる。
#
# また、同等となる式の構造が一意になるようにする。
# これによって同等の式のチェックが==による比較で行える。
# このために、AddSub、MulDiv ノードではオペランドをソートした状態で持つとする。
# (現在はノード側ではソートせず、与える側がソート済みであることを保証する)
#
# ※なお現状で重複生成が発生するケースは大きく分けて2種類ある。
#  1. 項が重複している場合。「1, 1, 3, 7」など。
#  2. 2つの部分式を異なる順序で生成する2つの探索ルートがある場合。
#     たとえば A, B, C, D の4つの数の組からは (A+B)+(C+D) という式が
#     2回生成される。先に A+B という部分式を生成する場合と
#     先に C+D という部分式を生成する場合である。

class SimpleExpr:
    """
    式を表す構文木の基本クラス。
    このサブクラスが表す構文木は次の条件を満たす。

     - 値のリテラルは Value ノードで表す。
     - eval メソッドで式の値を計算できる。
     - 等価な構文木を == で比較すると True になる。
     - 式を表すノードの表現は、値の順序や括弧の有無などを整理したときに
       等価になる式では同一になる。
       - 例: 3+(5+7) と (3+5)+7 と (5+7)+3 と 7+(3+5)
       - 例: 3-(5-7) と 3+(7-5) と (3-5)+7 (括弧を開けば 3+7-5 になる)
     - str()で式の文字列表現を得ることができる。そのとき不要な括弧はつかない。
    """
    def eval(self) -> float:
        """
        式の値を計算する。
        """
        raise NotImplemented

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self}>"

    def __str__(self) -> str:
        return self.to_s(head=True)

    def to_s(self, *, head: bool = False) -> str:
        """
        式を文字列化する。項が式の先頭の項であれば head には True を渡す。
        """
        return '()'


class Value (SimpleExpr):
    """
    値ノード。
    """
    def __init__(self, value: int) -> None:
        self.value = value

    def eval(self) -> float:
        return self.value

    def to_s(self, *, head: bool = False) -> str:
        if self.value < 0 and not head:
            return "(" + str(self.value) + ")"
        else:
            return str(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, Value):
            return self.value == other.value
        else:
            return False

    def __hash__(self) -> int:
        return self.value


T = TypeVar('T', bound=SimpleExpr)


def evalargs(args: Iterable[T]) -> Generator[float, None, None]:
    return (x.eval() for x in args)


def safeeval(x: SimpleExpr) -> float:
    try:
        return x.eval()
    except ZeroDivisionError:
        return 0


def mysorted(args, *, key):
    args = list(args)
    if len(args) < 2:
        return args
    else:
        return sorted(args, key=key)


def sortargs(args: Sequence[T]) -> List[T]:
    values = filter(lambda x: isinstance(x, Value), args)
    subexprs = filter(lambda x: not isinstance(x, Value), args)

    return mysorted(values, key=safeeval) + mysorted(subexprs, key=sortkey)


def sortkey(expr: SimpleExpr) -> List:
    """
    SimpleExprをソートするためのkey関数。
    """
    if isinstance(expr, Value):
        return [expr.value]
    elif isinstance(expr, AddSub):
        return [safeeval(expr),
                list(map(sortkey, expr.addargs)),
                list(map(sortkey, expr.subargs))]
    elif isinstance(expr, MulDiv):
        return [safeeval(expr),
                list(map(sortkey, expr.mulargs)),
                list(map(sortkey, expr.divisors))]
    else:
        raise TypeError('expr must be a SimpleExpr')


class AddSub (SimpleExpr):
    """
    加算・減算ノード。
    """

    def __init__(self, addargs: Iterable[Union[Value, MulDiv]], subargs: Iterable[Union[Value, MulDiv]]) -> None:
        """
        addargs, subargs は sortargs によりソート済みと仮定する。
        それによって、引数の組み合わせに対して内部構造が一意になる。
        """

        self.addargs: Tuple[Union[Value, MulDiv], ...] = tuple(addargs)
        self.subargs: Tuple[Union[Value, MulDiv], ...] = tuple(subargs)

        assert self.addargs, 'addargs must not be empty'
        self._eval :Optional[float] = None

    def eval(self) -> float:
        if self._eval is None:
            self._eval = add(evalargs(self.addargs)) - add(evalargs(self.subargs))
        return self._eval

    def to_s(self, *, head: bool = False) -> str:
        s1 = self.addargs[0].to_s(head=head)
        s2 = ''.join('+' + x.to_s(head=False) for x in self.addargs[1:])
        s3 = ''.join('-' + x.to_s(head=False) for x in self.subargs)
        return s1 + s2 + s3

    def __eq__(self, other) -> bool:
        if isinstance(other, AddSub):
            return self.addargs == other.addargs and self.subargs == other.subargs
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.addargs) * 13 + hash(self.subargs) * 47


class MulDiv (SimpleExpr):
    """
    乗算・除算ノード。
    """

    def __init__(self, mulargs: Iterable[Union[Value, AddSub]], divisors: Iterable[Union[Value, AddSub]]) -> None:
        """
        mulargs, divisors は sortargs によりソート済みと仮定する。
        それによって、引数の組み合わせに対して内部構造が一意になる。
        """

        self.mulargs: Tuple[Union[Value, AddSub], ...] = tuple(mulargs)
        self.divisors: Tuple[Union[Value, AddSub], ...] = tuple(divisors)

        assert self.mulargs, 'mulargs must not be empty'
        self._eval :Optional[float] = None

    def eval(self) -> float:
        if self._eval is None:
            self._eval = mul(evalargs(self.mulargs)) / mul(evalargs(self.divisors))
        return self._eval

    def _argstr(self, arg, head: bool = False) -> str:
        if isinstance(arg, AddSub):
            return "(" + arg.to_s(head=True) + ")"
        else:
            return arg.to_s(head=head)

    def to_s(self, *, head: bool = False) -> str:
        s1 = self._argstr(self.mulargs[0], head=head)
        s2 = ''.join('*' + self._argstr(x, head=False) for x in self.mulargs[1:])
        s3 = ''.join('/' + self._argstr(x, head=False) for x in self.divisors)
        return s1 + s2 + s3

    def __eq__(self, other) -> bool:
        if isinstance(other, MulDiv):
            return self.mulargs == other.mulargs and self.divisors == other.divisors
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.mulargs) * 31 + hash(self.divisors) * 23


def signs_to_str(signs):
    return ''.join(('+', '-')[s < 0] for s in signs)

###
### 演算子
###

# 型エイリアス
Operator = Callable[[Sequence[SimpleExpr]], SimpleExpr]
OpSign = Union[Literal[1], Literal[-1]]


def op_addsub(signs) -> Operator:
    """
    len(signs)個の項を引数とし、項番号に対応するsignsの値が正の項を加算、
    負の項を減算する演算子を返す。
    """
    def op(args) -> SimpleExpr:
        xs = [a for s, a in zip(signs, args) if s > 0]
        ys = [a for s, a in zip(signs, args) if s < 0]
        return AddSub(xs, ys)
    op.name = "addsub" + signs_to_str(signs) # type: ignore

    return op


def op_muldiv(signs) -> Operator:
    """
    len(signs)個の項を引数とし、項番号に対応するsignsの値が正の項を乗算、
    負の項を除算する演算子を返す。
    """
    def op(args) -> SimpleExpr:
        xs = [a for s, a in zip(signs, args) if s > 0]
        ys = [a for s, a in zip(signs, args) if s < 0]
        return MulDiv(xs, ys)
    op.name = "muldiv" + signs_to_str(signs) # type:ignore

    return op


# メモ化用記憶
_OPSIGNS: Dict[int, List[OpSign]] = {}


def opsigns(n: int) -> List[OpSign]:
    """
    opsigns は引数をポジティブな項とネガティブな項に
    振り分ける分け方を指定する。

     - ポジティブな項: AddSubの場合は加算、MulDivの場合は乗算する項
     - ネガティブな項: AddSubの場合は減算、MulDivの場合は除算する項

    ただしポジティブな項が0個になるような振り分け方はしないこととする。
    """
    if n not in _OPSIGNS:
        _OPSIGNS[n] = cast(List[OpSign], list(product((1, -1), repeat=n))[:-1])
    return _OPSIGNS[n]


# メモ化用記憶
_OPERATORS: Dict[Tuple[int, bool, bool], List[Callable[[Sequence[SimpleExpr]], SimpleExpr]]] = {}


def has_subtraction(expr: MulDiv) -> bool:
    """
    項に減算を含むなら真を返す。
    """
    return any((isinstance(t, AddSub) and t.subargs)
               for t in expr.mulargs + expr.divisors)


def single_exprs(terms: Sequence[SimpleExpr]) -> Generator[SimpleExpr, None, None]:
    """
    terms に含まれる項全てをオペランドとする演算ノードを生成する。
    """
    ops = operators(terms)

    for op in ops:
        yield op(terms)


def operators(terms: Sequence[SimpleExpr]) -> List[Operator]:
    """
    引数に対して可能な演算のリストを生成する。
    """
    arity = len(terms)
    can_addsub = AddSub not in map(type, terms)
    can_muldiv = MulDiv not in map(type, terms)
    ops: List[Operator] = []

    if can_addsub:
        # 項によっては加算のみ生成し、減算は生成しない。
        def skip_sub(expr):
            """
            exprを減数とする減算を生成しないならTrueを返す。
            """

            # X-(a*(b-c)) は X+(a*(c-b)) と同等なので、生成しない。
            #
            # ※ c-b が生成されない可能性があるが、大丈夫か?
            # 例: 5-(2*(0-3)) はスキップできるか?
            # 同等の式である 5+(2*(3-0)) は生成されない。
            # しかし、それと同等の式である 5+(2*(3+0)) は生成されるので
            # スキップしても大丈夫。
            #
            # 式を生成しないのは、あくまで同等の式がほかに生成されるからなので、
            # 部分式が生成されないためにスキップの可否が変更になることはない。
            if type(expr) == MulDiv and has_subtraction(expr):
                return True

            # -0 は +0 と同等なので、0(または0除算になる項) は減算を生成しない。
            if safeeval(expr) == 0:
                return True

            # それ以外は減算を生成する。
            return False

        skip_sub_mask = list(map(skip_sub, terms))

        # (検討中) 値が0になる AddSub は重複を省く。
        #
        # A-B+C の値が0ならば、符号を反転させた -A+B-C の値も0である。
        # 両者は同等の式と見做すので、片方だけ生成すればよい。
        #
        # よって生成した AddSub の値が0になる場合、最初の非0の項が
        # 加算項なら残し、減算項なら捨てる。

        def use_addsub(opsigns):
            for s, m in zip(opsigns, skip_sub_mask):
                if s < 0 and m:
                    return False
            return True
        ops += list(map(op_addsub, filter(use_addsub, opsigns(arity))))

    if can_muldiv:
        # 項によっては乗算のみ生成し、除算は生成しない。
        def skip_div(expr):
            """
            exprを除数とする除算を生成しないならTrueを返す。
            """
            # X/0 は生成しない。
            # また、X/±1 は X*±1 と同等なので、生成しない。
            if safeeval(expr) in (0, 1, -1):
                return True

            # それ以外は生成する。
            return False

        skip_div_mask = list(map(skip_div, terms))

        # (検討中) 減算を含む項が複数ある場合、特定のケースのみ生成する。
        #
        # A-B, C-D, E の3項を引数とする場合を考える。
        # 他に B-A, D-C, E という場合も(一部を除き)生成されるはずである。
        # その場合、(A-B)*(C-D)*E を生成すれば (B-A)*(D-C)*E は生成しなくてよい。
        #
        # また、A-B, C-D, E-F の3項の場合を考える。
        # それぞれの符号を考えると次の8通りが考えられる。
        # (A-Bを+, B-Aを-とする)
        # これらのうち、生成する必要があるのは
        # 1, 4, 6, 7 のうち1通り、2, 3, 5, 8のうち1通り、の合計2通りだけである。
        #
        # 1. +,+,+
        # 2. +,+,-
        # 3. +,-,+
        # 4. +,-,-
        # 5. -,+,+
        # 6. -,+,-
        # 7. -,-,+
        # 8. -,-,-
        #
        # 今 A-B, C-D, E-F の3項による MulDiv を構成しようとしているということは
        # 別の探索ルートで A-B, C-D, F-E や A-B, D-C, F-E など相当の3項による
        # MulDiv も構成しようとされるということである。
        #
        # 上記のように、それらの半分は同等なので、次のようにして生成を省略できる。
        # - 減算を含む項の値を掛け合わせて >0 になる場合:
        #   全ての減算項の値が正の場合のみ MulDiv を生成する。
        # - 減算を含む項の値を掛け合わせて <0 になる場合:
        #   最後の減算項の値が負、それ以外の減算項の値が正の場合のみ MulDiv を生成する。
        # ※上記の減算を含む項は、値が0になるものは含めずに考える。

        # (検討中) 0となる項がある場合、乗算のみを生成する。
        #
        # 0が除数になる式はもちろん生成しない(skip_divでも実現済)。
        #
        # 0が乗数(被除数)になる場合、他の項が乗数であれ除数であれ
        # 式の値は0になるから、
        # 0*A*B も 0*A/B も 0/A/B も同じと見做す。

        def use_muldiv(opsigns):
            for s, m in zip(opsigns, skip_div_mask):
                if s < 0 and m:
                    return False
            return True
        ops += list(map(op_muldiv, filter(use_muldiv, opsigns(arity))))

    return ops


# def operators(terms):
#     arity = len(terms)
#     can_addsub = AddSub not in map(type, terms)
#     can_muldiv = MulDiv not in map(type, terms)
#     return operators_1(len(terms),
#                        AddSub not in map(type, terms),
#                        MulDiv not in map(type, terms))


def operators_1(arity: int, can_addsub: bool, can_muldiv: bool) -> List[Callable[[Sequence[SimpleExpr]], SimpleExpr]]:
    """
    n個の引数に対して可能な演算のリストを生成する。

    AddSub、MulDivは同じクラスの項の直接の入れ子にならない。
    すなわち、
    引数にAddSubがある場合、op_addsubは演算子にならない。
    引数にMulDivがある場合、op_muldivは演算子にならない。

    もし引数に選ばれた項に AddSub と MulDiv の両方が含まれる場合は
    式は完成されない。
    """
    key = (arity, can_addsub, can_muldiv)
    if key in _OPERATORS:
        return _OPERATORS[key]

    ops: List[Callable[[Sequence[SimpleExpr]], SimpleExpr]] = []
    if can_addsub:
        ops += list(map(op_addsub, opsigns(arity)))
    if can_muldiv:
        ops += list(map(op_muldiv, opsigns(arity)))

    _OPERATORS[key] = ops
    return ops


###
### 式の構築
###

def build_exprs_1(terms: List[SimpleExpr]) -> Generator[SimpleExpr, None, None]:
    # terms はソートされていなければならない。

    if len(terms) == 1:
        try:
            LOG.debug("yield expr %s =%d", terms[0], terms[0].eval())
        except ZeroDivisionError:
            LOG.debug("yield expr %s =ERROR (divide by 0)", terms[0])
        yield terms[0]
        return

    # とりうる全ての構造の式を生成
    # ---------------------------
    #
    # 2個以上len(terms)個以下の項を選び、
    # それらを引数に演算を行う項で置き換える。
    # 再帰によりこれを繰り返して、項が一つになったら
    # それを式全体の構文木として出力する。

    dups_in_terms = len(terms) != len(set(terms))
    checked :Set[Tuple[SimpleExpr, ...]] = set()

    for arity in range(2, len(terms) + 1):
        for indices in combinations(range(len(terms)), arity):
            # indices は常に昇順になる。
            #
            # terms はソート済みと仮定するので、
            # selected_terms の要素の並びも要素の組に対して一意になる。
            #
            # それによって、これらを引数とする AddSub や MulDiv の
            # 内部構造も引数の組合せに対して一意になる。
            selected_terms = tuple(terms[i] for i in indices)

            if dups_in_terms:
                # 項に重複がある場合、取り出した項の組み合わせも
                # 重複が生じる。重複した場合は調べる手間が無駄なので省く。
                if selected_terms in checked:
                    continue
                checked.add(selected_terms)

            # (検討中) 4つ組 (A, B, C, D) の場合、
            # ((A, B), (C, D)) と ((C, D), (A, B)) の2通りを
            # 生成してしまう。
            # 条件をつけてこれを抑制できないか?

            rest: List[SimpleExpr] = terms.copy()
            for i in reversed(indices): # indicesが昇順であることを前提とする
                del rest[i]

            LOG.debug(f"choose %d numbers: indices=%s values=%s",
                      arity,
                      indices,
                      ','.join(map(str, selected_terms)))

            for expr in single_exprs(selected_terms):
                yield from build_exprs_1(sortargs([expr] + rest))


def build_exprs(nums: Sequence[float]) -> Generator[SimpleExpr, None, None]:
    """
    numsで与えられた項からなる式を全ての構造で生成する。
    """
    values :List[SimpleExpr] = list(map(Value, sorted(nums))) # type: ignore
    yield from build_exprs_1(values)

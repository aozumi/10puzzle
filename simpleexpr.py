from __future__ import annotations
from functools import reduce
from typing import Union, List, Literal, Sequence, Generator, Iterable, TypeVar, cast, Optional, Callable, Tuple, Dict, Set
from itertools import combinations, product
import logging

from utils import first, subsums


LOG = logging.getLogger(__name__)


def prod(args: Iterable[float]) -> float:
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

    def minid(self) -> int:
        raise NotImplementedError()

    def maxid(self) -> int:
        raise NotImplementedError()

    def eval(self) -> float:
        """
        式の値を計算する。
        """
        raise NotImplementedError()

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
    id_counter = 0

    @classmethod
    def next_id(klass) -> int:
        r = Value.id_counter
        Value.id_counter += 1
        return r

    @classmethod
    def reset_id(klass) -> None:
        Value.id_counter = 0


    """
    値ノード。
    """
    def __init__(self, value: int) -> None:
        self.value = value
        self.id = Value.next_id()

    def minid(self) -> int:
        return self.id

    def maxid(self) -> int:
        return self.id

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
S = TypeVar('S')


def eval_exprs(args: Iterable[T]) -> Generator[float, None, None]:
    return (x.eval() for x in args)


def eval_expr(x: SimpleExpr) -> float:
    return x.eval()


def mysorted(args: Iterable[T], *, key) -> List[T]:
    args = list(args)
    if len(args) < 2:
        return args
    else:
        return sorted(args, key=key)


def sort_1(value_key: Callable[[Value], S],
           subexpr_key: Callable[[Union[AddSub, MulDiv]], S],
           exprs: Iterable[T]
) -> Sequence[T]:
    if isinstance(exprs, Sequence) and len(exprs) < 2:
        return exprs

    exprs_list = list(exprs)
    if len(exprs_list) < 2:
        return exprs_list

    values = filter(lambda x: isinstance(x, Value), exprs_list)
    subexprs = filter(lambda x: not isinstance(x, Value), exprs_list)

    return mysorted(values, key=value_key) + mysorted(subexprs, key=subexpr_key)


def sort_by_value(exprs: Iterable[T]) -> Sequence[T]:
    return sort_1(eval_expr, sort_by_value_key, exprs)


def sort_by_value_key(expr: SimpleExpr) -> List:
    """
    SimpleExprをソートするためのkey関数。
    """
    if isinstance(expr, Value):
        return [expr.value]
    elif isinstance(expr, AddSub):
        return [expr.eval(),
                list(map(sort_by_value_key, expr.addargs)),
                list(map(sort_by_value_key, expr.subargs))]
    elif isinstance(expr, MulDiv):
        return [expr.eval(),
                list(map(sort_by_value_key, expr.mulargs)),
                list(map(sort_by_value_key, expr.divisors))]
    else:
        raise TypeError('expr must be a SimpleExpr')


def sort_by_id(exprs: Iterable[T]) -> Sequence[T]:
    minid = lambda x: x.minid()
    return sort_1(minid, minid, exprs)


class AddSub (SimpleExpr):
    """
    加算・減算ノード。
    """

    def __init__(self, addargs: Iterable[Union[Value, MulDiv]], subargs: Iterable[Union[Value, MulDiv]]) -> None:
        # addargs, subargs は値でソートして保持する。
        # それによって、引数の組み合わせに対して内部構造が一意になる。
        #
        # また、2*3+2*5 のように同じ数を複数含む構造も一意に定まる。
        # (idでソートする場合、AddSub([2*3, 2'*5], []) と
        # AddSub([2*5, 2'*3], []) の2通りが生じてしまう)
        self.addargs: Tuple[Union[Value, MulDiv], ...] = tuple(sort_by_value(addargs))
        self.subargs: Tuple[Union[Value, MulDiv], ...] = tuple(sort_by_value(subargs))

        assert self.addargs, 'addargs must not be empty'
        self._eval :Optional[float] = None

    def minid(self) -> int:
        return min(x.minid() for x in (self.addargs + self.subargs))

    def maxid(self) -> int:
        return max(x.maxid() for x in (self.addargs + self.subargs))

    def eval(self) -> float:
        if self._eval is None:
            self._eval = sum(eval_exprs(self.addargs)) - sum(eval_exprs(self.subargs))
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
        # mulargs, divisors は値でソートして保持する。
        # それによって、引数の組み合わせに対して内部構造が一意になる。
        self.mulargs: Tuple[Union[Value, AddSub], ...] = tuple(sort_by_value(mulargs))
        self.divisors: Tuple[Union[Value, AddSub], ...] = tuple(sort_by_value(divisors))

        assert self.mulargs, 'mulargs must not be empty'
        self._eval :Optional[float] = None

    def minid(self) -> int:
        return min(x.minid() for x in (self.mulargs + self.divisors))

    def maxid(self) -> int:
        return max(x.maxid() for x in (self.mulargs + self.divisors))

    def eval(self) -> float:
        if self._eval is None:
            self._eval = prod(eval_exprs(self.mulargs)) / prod(eval_exprs(self.divisors))
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
_OPSIGNS: Dict[int, List[Tuple[OpSign, ...]]] = {}


def opsigns_full(n: int) -> List[Tuple[OpSign, ...]]:
    """
    opsigns は引数をポジティブな項とネガティブな項に
    振り分ける分け方を指定する。

     - ポジティブな項: AddSubの場合は加算、MulDivの場合は乗算する項
     - ネガティブな項: AddSubの場合は減算、MulDivの場合は除算する項

    ただしポジティブな項が0個になるような振り分け方はしないこととする。
    """
    if n not in _OPSIGNS:
        _OPSIGNS[n] = cast(List[Tuple[OpSign, ...]], list(product((1, -1), repeat=n)))[:-1]
    return _OPSIGNS[n]


def has_subtraction(expr: MulDiv) -> bool:
    """
    項に減算を含むなら真を返す。
    """
    return any((isinstance(t, AddSub) and t.subargs)
               for t in expr.mulargs + expr.divisors)


def opsigns(skip_negative_p, terms) -> Iterable[Tuple[OpSign, ...]]:
    """
    opsigns は引数をポジティブな項とネガティブな項に
    振り分ける分け方を指定する。

     - ポジティブな項: AddSubの場合は加算、MulDivの場合は乗算する項
     - ネガティブな項: AddSubの場合は減算、MulDivの場合は除算する項

    ただしポジティブな項が0個になるような振り分け方はしないこととする。
    また、skip_negative_p が真を返す項はネガティブな項にはしない。
    """
    skips = list(map(skip_negative_p, terms))
    POS_NEG = (1, -1)
    POS_ONLY = (1,)
    if any(skips):
        return product(*((POS_NEG, POS_ONLY)[s] for s in skips))
    else:
        return opsigns_full(len(terms))


def single_exprs(terms: Sequence[SimpleExpr]) -> Generator[SimpleExpr, None, None]:
    """
    terms に含まれる項全てをオペランドとする演算ノードを生成する。
    """
    # termsはidでソートされていることを仮定する。

    arity = len(terms)
    can_addsub = AddSub not in map(type, terms)
    can_muldiv = MulDiv not in map(type, terms)

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
            if expr.eval() == 0:
                return True

            # それ以外は減算を生成する。
            return False

        # 非0項のインデックスのリスト
        nonzero_indices = [i for i,t in enumerate(terms) if t.eval() != 0]

        # 最初の非0項のインデックス (非0項がなければ None)
        first_nonzero_index = nonzero_indices[0] if nonzero_indices else None

        for s in opsigns(skip_sub, terms):
            expr = op_addsub(s)(terms)

            # 値が0になる AddSub は重複を省く。
            #
            # A-B+C の値が0ならば、符号を反転させた -A+B-C の値も0である。
            # 両者は同等の式と見做すので、片方だけ生成すればよい。
            #
            # よって生成した AddSub の値が0になる場合、最初の非0の項を
            # 加えるなら残し、減ずるなら捨てる。
            #
            # この処理には、terms がソートされていることは必要ではない。
            if (expr.eval() == 0
                and first_nonzero_index is not None
                and s[first_nonzero_index] < 0):
                continue

            # 任意個の部分項(各項は非0)の和が0になる場合、
            # それらを符号反転したケースも0になる。
            # よって両者を重複するとみなし片方を排除したい。
            # 最初の項が加算項なら残し、減算項なら捨てる。
            #
            # つまり、先頭のsが負で部分和が0になる非0項の部分集合があるなら
            # 生成しない。
            #
            # これは上の重複排除を一般化したもの。

            nonzero_values = [terms[i].eval() * s[i] for i in nonzero_indices]

            if any(-nonzero_values[j] in subsums(nonzero_values[j+1:])
                   for j in range(len(nonzero_indices) - 1)
                   if s[nonzero_indices[j]] < 0):
                continue

            yield expr

    if can_muldiv:
        # 項によっては乗算のみ生成し、除算は生成しない。
        def skip_div(expr):
            """
            exprを除数とする除算を生成しないならTrueを返す。
            """
            # X/0 は生成しない。
            # また、X/±1 は X*±1 と同等なので、生成しない。
            if expr.eval() in (0, 1, -1):
                return True

            # それ以外は生成する。
            return False

        # 0となる項がある場合、乗算のみを生成する。
        #
        # 0が除数になる式はもちろん生成しない(skip_divでも実現済)。
        #
        # 0が乗数(被除数)になる場合、他の項が乗数であれ除数であれ
        # 式の値は0になるから、
        # 0*A*B も 0*A/B も 0/A/B も同じと見做す。

        if 0 in eval_exprs(terms):
            yield op_muldiv([1]*arity)(terms)
            return

        # 以下では、全ての項は非0である

        # 減算を含む項が複数ある場合、特定のケースのみ生成する。
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
        #   例: 合計1, 3, 5, 7で4にする場合の (1-3)*(7-5)と(5-7)*(3-1)
        #   最後の減算項の値が負、それ以外の減算項の値が正の場合のみ MulDiv を
        #   生成する。
        #   ただし項の順序が値によって変化しないことが必要。
        #   (符号が変われば値が変わるから)
        #
        # ※上記の減算を含む項は、値が0になるものは含めずに考える。

        nonzero_subtractions = [ t.eval() for t in terms
                                 if isinstance(t, AddSub) and t.subargs
        ]

        if len(nonzero_subtractions) >= 2:
            if prod(nonzero_subtractions) > 0:
                # 減算を含む項が複数あり、掛け合わせて正になる場合、
                # 全ての項の値が正の場合のみ生成する。
                # その他の場合は重複ケースとみなして捨てる。
                if not all(x > 0 for x in nonzero_subtractions):
                    return

            else:
                # 減算を含む項が複数あり、掛け合わせて負になる場合、
                # 最後の項を除く全ての項の値が正の場合のみ生成する。
                # ただしソートが値によらない(idでのソート)ことを前提とする。
                # その他の場合は重複ケースとみなして捨てる。
                if not all(x > 0 for x in nonzero_subtractions[:-1]):
                    return

        for s in opsigns(skip_div, terms):
            yield op_muldiv(s)(terms)


###
### 式の構築
###

def build_exprs_1(terms: Sequence[SimpleExpr]) -> Generator[SimpleExpr, None, None]:
    # termsはidでソートされていることを仮定する。

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

            rest: List[SimpleExpr] = list(terms)
            for i in reversed(indices): # indicesが昇順であることを前提とする
                del rest[i]

            LOG.debug(f"choose %d numbers: indices=%s values=%s",
                      arity,
                      indices,
                      ','.join(map(str, selected_terms)))

            for expr in single_exprs(selected_terms):
                yield from build_exprs_1(sort_by_id([expr] + rest))


def build_exprs(nums: Sequence[float]) -> Generator[SimpleExpr, None, None]:
    """
    numsで与えられた項からなる式を全ての構造で生成する。
    """
    # values は値でもidでも昇順に並んでいる
    values :List[SimpleExpr] = list(map(Value, sorted(nums))) # type: ignore
    yield from build_exprs_1(values)

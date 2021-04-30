from typing import Generator, Iterable, Sequence, TypeVar, Union, cast
from itertools import combinations


T = TypeVar('T')
S = TypeVar('S')


def first(xs: Iterable[T], default: S = None) -> Union[T, S]:
    "xs の最初の要素を返す。xs が空の場合は default を返す。"
    return next(iter(xs), cast(S, default))


def subsums(values: Sequence[T]) -> Generator[T, None, None]:
    """
    valuesの部分和を全て生成する。
    """
    for size in range(1, len(values) + 1):
        for subset in combinations(values, size):
            # sum: Union[T, int] だが、int は引数が空のケース(0を返す)。
            # subset は空ではないので、T にキャストして問題ない。
            yield cast(T, sum(subset))

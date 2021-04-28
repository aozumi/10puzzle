from typing import Iterable, TypeVar, Union, cast


T = TypeVar('T')
S = TypeVar('S')


def first(xs: Iterable[T], default: S = None) -> Union[T, S]:
    "xs の最初の要素を返す。xs が空の場合は default を返す。"
    return next(iter(xs), cast(S, default))

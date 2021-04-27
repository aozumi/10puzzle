#!/usr/local/bin/python3
#
# 4つの値 1, 1, 5, 8 と四則演算と括弧を用いて、答が10になる式を作れ。
#
# - 演算は四則演算のみ。べき乗、平方根などは不可。
# - 数字を並べて合体させる(例: 1 と 1 で 11)のは不可。

from typing import Sequence, Generator, Iterable, List, Optional, Set
import logging
import sys

from simpleexpr import SimpleExpr, build_exprs

LOG = logging.getLogger(__name__)


def remove_duplicated_exprs(exprs: Iterable[SimpleExpr]) -> Generator[SimpleExpr, None, None]:
    seen: Set[SimpleExpr] = set()
    for x in exprs:
        if x in seen:
            continue
        seen.add(x)
        yield x


def find_solutions(numbers: Sequence[float], value=10) -> Generator[SimpleExpr, None, None]:
    n = 0
    for expr in build_exprs(numbers):
        n += 1
        try:
            result = expr.eval()
            if result == int(result):
                LOG.debug("%s=%d", expr, result)
            else:
                LOG.debug("%s=%f", expr, result)
            if result == value:
                yield expr
        except ZeroDivisionError:
            LOG.debug("%s=ERROR (divide by 0)", expr)
    LOG.debug("%d exprs generated", n)


QUIZ1 = [1, 1, 5, 8]
QUIZ2 = [3, 4, 7, 8]
QUIZ3 = [9, 9, 9, 9]


def init_options() -> Optional[List[int]]:
    args = sys.argv[1:]

    if len(args) == 0:
        return None

    if len(args) >= 2:
        return list(map(int, args))

    print("usage: puzzle.py [N N ...]")
    exit(1)


def main() -> None:
    logging.basicConfig(
        # level=logging.DEBUG,
    )

    q = init_options()
    if q:
        qs = [q]
    else:
        qs = [QUIZ1, QUIZ2, QUIZ3]

    for quiz in qs:
        print("")
        print("Quiz: " + ', '.join(map(str, quiz)))

        found = False
        for sol in remove_duplicated_exprs(find_solutions(quiz)):
            if not found:
                print("Solutions:")
            found = True
            print("  " + str(sol))
        if not found:
            print("No solution found.")
            exit(3)


if __name__ == '__main__':
    main()

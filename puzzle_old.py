#!/usr/local/bin/python3
#
# 4つの値 1, 1, 5, 8 と四則演算と括弧を用いて、答が10になる式を作れ。
#
# - 演算は四則演算のみ。べき乗、平方根などは不可。
# - 数字を並べて合体させる(例: 1 と 1 で 11)のは不可。

from itertools import permutations, product
from typing import Sequence, Generator
import logging
import sys

from expr import build_exprs, Expr, OPERATORS, remove_duplicated_exprs

LOG = logging.getLogger(__name__)


def find_solutions(numbers: Sequence[float]) -> Generator[Expr, None, None]:
    for nums in permutations(numbers):
        for ops in product(OPERATORS, repeat=len(numbers) - 1):
            for expr in build_exprs(nums, ops):
                try:
                    result = expr.eval()
                    LOG.debug("%s=%d", expr, result)
                    if result == 10:
                        yield expr
                except ZeroDivisionError:
                    LOG.debug("%s=ERROR (divide by 0)", expr)


QUIZ1 = [1, 1, 5, 8]
QUIZ2 = [3, 4, 7, 8]
QUIZ3 = [9, 9, 9, 9]


def init_options():
    args = sys.argv[1:]

    if len(args) == 0:
        return None

    if len(args) >= 2:
        return list(map(int, args))

    print("usage: puzzle.py [N N ...]")
    exit(1)


def main():
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

        solutions = remove_duplicated_exprs(find_solutions(quiz))
        if solutions:
            print("Solutions:")
            for sol in solutions:
                print("  " + str(sol))
        else:
            print("No solution found.")


if __name__ == '__main__':
    main()

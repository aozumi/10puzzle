#!/usr/local/bin/python3
#
# 4つの値 1, 1, 5, 8 と四則演算と括弧を用いて、答が10になる式を作れ。
#
# - 演算は四則演算のみ。べき乗、平方根などは不可。
# - 数字を並べて合体させる(例: 1 と 1 で 11)のは不可。

import argparse
from typing import Sequence, Generator, Iterable, List, Optional, Set
import logging
import sys
from dataclasses import dataclass

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


@dataclass
class Options:
    numbers: List[int]
    value: int


def init_options() -> Optional[List[int]]:
    parser = argparse.ArgumentParser(description='10パズルの解を探す')
    parser.add_argument('--value', metavar='N',
                        help='解となる式の値 (デフォルト: 10)',
                        type=int,
                        default=10)
    parser.add_argument('numbers', metavar='N',
                        type=int,
                        nargs='+')

    args = parser.parse_args()
    return Options(args.numbers, args.value)


def main() -> None:
    logging.basicConfig(
        # level=logging.DEBUG,
    )

    options = init_options()
    quiz = sorted(options.numbers)

    found = False
    for sol in remove_duplicated_exprs(find_solutions(quiz, options.value)):
        found = True
        print(sol)
    if not found:
        print("No solution found.", file=sys.stderr)
        exit(3)


if __name__ == '__main__':
    main()

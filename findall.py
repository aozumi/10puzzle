#!/usr/local/bin/python3
#
# 指定された個数の数のリストで、10パズルの解を持つものを全て調べる。
# 最後に、解を持つ組み合わせの数を表示する。
#
# - 演算は四則演算のみ。べき乗、平方根などは不可。
# - 数字を並べて合体させる(例: 1 と 1 で 11)のは不可。

from itertools import permutations, product, combinations_with_replacement
from typing import Sequence, Generator, Iterable, List, Optional, Tuple
import logging
import sys
import argparse
from dataclasses import dataclass

from puzzle import find_solutions, remove_duplicated_exprs


LOG = logging.getLogger(__name__)


@dataclass
class Options:
    arity: int
    show_count: bool
    show_exprs: bool
    min_number: int
    max_number: int
    value: int


def init_options() -> Options:
    parser = argparse.ArgumentParser(
        description='10パズルの解を全て調べる'
    )
    parser.add_argument('--count',
                        help='最後に解を持つ数字の組の個数を表示する',
                        action='store_const',
                        const=True,
                        default=False)
    parser.add_argument('--expr',
                        help='見つけた式を全て表示する',
                        action='store_const',
                        const=True,
                        default=False)
    parser.add_argument('--min', metavar='N',
                        help='数の最小値 (デフォルト: 0)',
                        type=int,
                        default=0)
    parser.add_argument('--max', metavar='N',
                        help='数の最大値 (デフォルト: 9)',
                        type=int,
                        default=9)
    parser.add_argument('--value', metavar='N',
                        help='解となる式の値 (デフォルト: 10)',
                        type=int,
                        default=10)
    parser.add_argument('N',
                       help='数字の個数')

    args = parser.parse_args()
    return Options(int(args.N), args.count, args.expr,
                   args.min, args.max, args.value)


def main() -> None:
    logging.basicConfig(
        # level=logging.DEBUG,
    )

    options = init_options()

    digits = list(range(options.min_number, options.max_number + 1))

    found = 0
    for quiz in combinations_with_replacement(digits, options.arity):
        quiz_str = ",".join(map(str, quiz))
        if options.show_exprs:
            # 全ての解を出力する
            for expr in remove_duplicated_exprs(find_solutions(quiz, options.value)):
                print(f"{quiz_str}: {expr}")
                found += 1
        else:
            # 解の有無のみを調べる
            try:
                next(find_solutions(quiz))
                found += 1
                print(quiz_str)
            except StopIteration:
                pass

    if options.show_count:
        print("")
        print(found)


if __name__ == '__main__':
    main()

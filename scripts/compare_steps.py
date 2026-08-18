#!/usr/bin/env python
"""
Step 1 vs step 2, seed by seed, for a whole split family.

Reads the same results.json files summarise.py reads, so it needs nothing but
the experiments/ tree and can be run while seeds are still going -- a seed that
only one step has finished shows up as n/a on the other side rather than being
dropped, so it is obvious what is still missing.

The mean delta is over the seeds *both* steps have finished. Averaging step 2
over three seeds against step 1 over five would compare two different sets of
splits and quietly flatter whichever step ran the easier ones.

Usage:
    python scripts/compare_steps.py --split random
    python scripts/compare_steps.py --split random --dataset esol
    python scripts/compare_steps.py --split scaffold --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections.abc import Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.summarise import read_seed_results, seed_scores  # noqa: E402
from src.data import DATASET_NAMES, get_dataset_info  # noqa: E402
from src.data.datasets import (  # noqa: E402
    DEFAULT_SPLIT,
    OUTPUT_DIR,
    SPLIT_TYPES,
)

#: Metrics where a larger number is the better one. Everything else (rmse,
#: mae) improves downwards.
HIGHER_IS_BETTER = {'auc', 'acc', 'r2'}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='compare_steps.py',
        description='Step 1 vs step 2 over the seeds of one split family.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--split', default=DEFAULT_SPLIT, choices=SPLIT_TYPES,
                        help='which split family to compare')
    parser.add_argument('--dataset', default='all',
                        choices=['all'] + DATASET_NAMES,
                        help="dataset key, or 'all' for every registered one")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='restrict to these split seeds '
                             '(default: all found)')
    return parser


#: One score column, wide enough for the '-12.3456' a badly scaled target can
#: produce and for the header above it.
COLUMN = 13


def _cell(value: float | None) -> str:
    return f'{value:.4f}' if value is not None else 'n/a'


def _delta(step1: float | None, step2: float | None,
           higher_is_better: bool) -> tuple[float | None, str]:
    """Step 2 minus step 1, plus which way that points."""
    if step1 is None or step2 is None:
        return None, ''
    delta = step2 - step1
    if delta == 0:
        return delta, 'same'
    improved = delta > 0 if higher_is_better else delta < 0
    return delta, 'better' if improved else 'worse'


def compare(split: str, dataset: str,
            seeds: Sequence[int] | None = None) -> bool:
    """Print one dataset's table. False if neither step has run a seed."""
    step1 = read_seed_results(split, 1, dataset, seeds)
    step2 = read_seed_results(split, 2, dataset, seeds)
    if not step1 and not step2:
        return False

    metric = get_dataset_info(dataset)['metric']
    higher_is_better = metric in HIGHER_IS_BETTER
    direction = 'higher' if higher_is_better else 'lower'

    print(f'\n{split} / {dataset} -- {metric}, {direction} is better')
    header = ''.join(f'{name:>{COLUMN}}' for name in
                     ('step1 valid', 'step1 test', 'step2 valid',
                      'step2 test', 'delta test'))
    print(f'  {"seed":>4}{header}')

    paired: list[tuple[float, float]] = []
    per_step: dict[int, list[float]] = {1: [], 2: []}
    for seed in sorted(set(step1) | set(step2)):
        _, valid1, test1 = (seed_scores(1, step1[seed]) if seed in step1
                            else (metric, None, None))
        _, valid2, test2 = (seed_scores(2, step2[seed]) if seed in step2
                            else (metric, None, None))
        delta, verdict = _delta(test1, test2, higher_is_better)
        if delta is not None:
            paired.append((test1, test2))
        for step, test in ((1, test1), (2, test2)):
            if test is not None:
                per_step[step].append(test)
        cells = ''.join(f'{_cell(value):>{COLUMN}}' for value in
                        (valid1, test1, valid2, test2))
        d = f'{delta:+.4f}' if delta is not None else 'n/a'
        print(f'  {seed:>4}{cells}{d:>{COLUMN}}  {verdict}'.rstrip())

    for step in (1, 2):
        values = per_step[step]
        if not values:
            continue
        # Sample std, and none at all for a single seed -- 0.0000 there would
        # read like perfect agreement between seeds that were never compared.
        std = statistics.stdev(values) if len(values) > 1 else None
        spread = f' +/- {std:.4f}' if std is not None else ''
        print(f'  step{step} test mean over {len(values)} seed(s): '
              f'{statistics.fmean(values):.4f}{spread}')

    if paired:
        deltas = [b - a for a, b in paired]
        mean_delta = statistics.fmean(deltas)
        _, verdict = _delta(0.0, mean_delta, higher_is_better)
        wins = sum(1 for d in deltas
                   if (d > 0) == higher_is_better and d != 0)
        print(f'  delta over the {len(paired)} seed(s) both steps finished: '
              f'{mean_delta:+.4f} {verdict}   '
              f'({wins}/{len(paired)} seeds better)')
    else:
        print('  no seed has both steps finished yet -- no delta')
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    datasets = DATASET_NAMES if args.dataset == 'all' else [args.dataset]
    found = sum(compare(args.split, dataset, args.seeds)
                for dataset in datasets)
    if not found:
        print(f'\nNo {args.split} results under {OUTPUT_DIR}/{args.split}/')
        return 1
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())

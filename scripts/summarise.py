#!/usr/bin/env python
"""
Mean +/- std over the seeds of one experiment family.

Reads the results.json each run already writes, so it can be run at any point
and needs nothing but the experiments/ tree. Where a seed has been run more
than once, the newest run wins.

run_step1.py calls this at the end of every run, so the five-seed mean appears
on the terminal as soon as the fifth seed finishes.

Usage:
    python scripts/summarise.py --split random --step 1
    python scripts/summarise.py --split random --step 2 --dataset esol
    python scripts/summarise.py --split scaffold --step 1 --dataset lipo
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Sequence
from glob import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data import DATASET_NAMES  # noqa: E402
from src.data.datasets import (  # noqa: E402
    DEFAULT_SPLIT,
    OUTPUT_DIR,
    SPLIT_TYPES,
    dataset_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='summarise.py',
        description='Mean +/- std over the seeds of one experiment family.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--split', default=DEFAULT_SPLIT, choices=SPLIT_TYPES,
                        help='which split family to summarise')
    parser.add_argument('--step', type=int, default=1, choices=(1, 2),
                        help='which step wrote the runs')
    parser.add_argument('--dataset', default='all',
                        choices=['all'] + DATASET_NAMES,
                        help="dataset key, or 'all' for every registered one")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='restrict to these split seeds (default: all found)')
    return parser


def read_seed_results(split: str, step: int, dataset: str,
                      seeds: Sequence[int] | None = None
                      ) -> dict[int, dict]:
    """Newest results.json per seed, keyed by split seed."""
    root = os.path.join(OUTPUT_DIR, dataset_dir(split, f'step{step}', dataset))

    found: dict[int, dict] = {}
    for seed_dir in sorted(glob(os.path.join(root, 'seed_*'))):
        try:
            seed = int(os.path.basename(seed_dir).rsplit('_', 1)[1])
        except ValueError:
            continue
        if seeds is not None and seed not in seeds:
            continue
        runs = glob(os.path.join(seed_dir, '*', 'results.json'))
        if not runs:
            continue
        with open(max(runs, key=os.path.getmtime)) as fh:
            found[seed] = json.load(fh)
    return found


def seed_scores(step: int, result: dict) -> tuple[str, float | None, float | None]:
    """(metric, valid, test) for one run, whichever step wrote it."""
    metric = result.get('metric', 'rmse')
    if step == 1:
        # Step 1 keys its scores by split then metric name: {'test': {'rmse': x}}.
        valid = (result.get('valid') or {}).get(metric)
        test = (result.get('test') or {}).get(metric)
        return metric, valid, test

    # Step 2 reports the valid-selected phase. valid_score/test_score are
    # metric-agnostic; the rmse keys are what runs made before classification
    # support wrote.
    best = result.get('best') or {}
    valid = best.get('valid_score', best.get('valid_rmse'))
    test = best.get('test_score', best.get('test_rmse'))
    return metric, valid, test


def summarise(split: str, step: int, dataset: str,
              seeds: Sequence[int] | None = None) -> dict | None:
    """Print the per-seed table and the mean; return the aggregate, or None."""
    results = read_seed_results(split, step, dataset, seeds)
    if not results:
        return None

    metric = 'rmse'
    rows: list[tuple[int, float | None, float | None]] = []
    for seed in sorted(results):
        metric, valid, test = seed_scores(step, results[seed])
        rows.append((seed, valid, test))

    print(f'\n{split} / step{step} / {dataset} '
          f'-- {len(rows)} seed(s), {metric}')
    for seed, valid, test in rows:
        v = f'{valid:.4f}' if valid is not None else '  n/a '
        t = f'{test:.4f}' if test is not None else '  n/a '
        print(f'  seed {seed}   valid {v}   test {t}')

    aggregate = {'split': split, 'step': step, 'dataset': dataset,
                 'metric': metric, 'n_seeds': len(rows)}
    parts = []
    for name, index in (('valid', 1), ('test', 2)):
        values = [row[index] for row in rows if row[index] is not None]
        if not values:
            continue
        mean = statistics.fmean(values)
        # Sample std: these seeds are a sample of the split distribution, not
        # the whole of it. Undefined for a single seed -- report no spread at
        # all there rather than a 0.0000 that reads like perfect agreement.
        std = statistics.stdev(values) if len(values) > 1 else None
        aggregate[f'{name}_mean'], aggregate[f'{name}_std'] = mean, std
        parts.append(f'{name} {mean:.4f}'
                     + (f' +/- {std:.4f}' if std is not None else ''))
    if parts:
        print(f'  mean     {"   ".join(parts)}')
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    datasets = DATASET_NAMES if args.dataset == 'all' else [args.dataset]
    found = 0
    for dataset in datasets:
        if summarise(args.split, args.step, dataset, args.seeds) is not None:
            found += 1

    if not found:
        print(f'\nNo {args.split} step{args.step} results under '
              f'{OUTPUT_DIR}/{args.split}/step{args.step}/')
        return 1
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())

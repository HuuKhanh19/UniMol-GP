#!/usr/bin/env python
"""
Preprocess: clean raw CSVs and write Bemis-Murcko scaffold splits.

All configuration lives in argparse, like run_step1.py. There is no config file.

Output: data/processed/{dataset}/seed_{n}/{dataset}_{train,valid,test}.csv

Usage:
    python scripts/preprocess_data.py --dataset esol
    python scripts/preprocess_data.py --dataset esol --split-seed 0 1 2 3 4
    python scripts/preprocess_data.py --dataset all --split-seed 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data import DATASET_NAMES, prepare_dataset  # noqa: E402
from src.data.datasets import PROCESSED_DIR, RAW_DIR, SPLIT_RATIO  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='preprocess_data.py',
        description='Clean raw CSVs and write scaffold splits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', default='all',
                        choices=['all'] + DATASET_NAMES,
                        help="dataset key, or 'all' for every registered one")
    parser.add_argument('--split-seed', type=int, nargs='+', default=[0],
                        help='one or more scaffold-split seeds')
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line. Defaults come from build_parser(), nowhere else."""
    return build_parser().parse_args(argv)


def preprocess(dataset_name: str, split_seed: int) -> tuple[int, int, int]:
    """Split one dataset at one seed and write the three CSVs."""
    train_df, valid_df, test_df, _ = prepare_dataset(
        dataset_name, raw_dir=RAW_DIR,
        split_ratio=SPLIT_RATIO, split_seed=split_seed,
    )
    frames = (('train', train_df), ('valid', valid_df), ('test', test_df))

    out_dir = os.path.join(PROCESSED_DIR, dataset_name, f'seed_{split_seed}')
    os.makedirs(out_dir, exist_ok=True)
    for name, df in frames:
        df.to_csv(os.path.join(out_dir, f'{dataset_name}_{name}.csv'),
                  index=False)

    print(f'  Saved -> {out_dir}/')
    return len(train_df), len(valid_df), len(test_df)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)

    datasets: list[str] = (DATASET_NAMES if args.dataset == 'all'
                           else [args.dataset])
    print(f'\nPreprocessing | datasets={datasets} | seeds={args.split_seed}')
    print('=' * 60)

    failed = 0
    for dataset in datasets:
        for seed in args.split_seed:
            print(f'\n{dataset.upper()} (split_seed={seed})')
            try:
                n_train, n_valid, n_test = preprocess(dataset, seed)
            except FileNotFoundError as exc:
                print(f'  Skipped: {exc}')
                failed += 1
                continue
            print(f'  Train={n_train}, Valid={n_valid}, Test={n_test}')

    print(f"\n{'=' * 60}\nDone." + (f' ({failed} skipped)' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
"""
Preprocess: scaffold-split raw data into train/valid/test.

Usage:
    python scripts/preprocess_data.py --dataset esol --split-seed 0
    python scripts/preprocess_data.py --dataset esol --split-seed 0 1 2 3 4
    python scripts/preprocess_data.py --dataset all --split-seed 0 1 2 3 4
"""

import os, sys, argparse, yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import DATASET_NAMES, prepare_dataset
from src.data.datasets import RAW_DIR, PROCESSED_DIR, SPLIT_RATIO


def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def preprocess(dataset_name, split_seed):
    train_df, valid_df, test_df, _ = prepare_dataset(
        dataset_name, raw_dir=RAW_DIR,
        split_ratio=SPLIT_RATIO, split_seed=split_seed,
    )
    # Save to: data/processed/{dataset}/seed_{X}/
    out_dir = os.path.join(PROCESSED_DIR, dataset_name, f"seed_{split_seed}")
    os.makedirs(out_dir, exist_ok=True)
    for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        df.to_csv(os.path.join(out_dir, f"{dataset_name}_{name}.csv"), index=False)
    print(f"  Saved → {out_dir}/")
    return len(train_df), len(valid_df), len(test_df)


def main():
    parser = argparse.ArgumentParser(description="Preprocess molecular datasets")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all'] + DATASET_NAMES)
    parser.add_argument('--split-seed', type=int, nargs='+', default=None)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    os.chdir(project_root)

    cfg = load_config(args.config)
    seeds = args.split_seed if args.split_seed is not None else [cfg.get('split_seed', 0)]
    datasets = DATASET_NAMES if args.dataset == 'all' else [args.dataset]

    print(f"\nPreprocessing | datasets={datasets} | seeds={seeds}")
    print("=" * 60)
    for ds in datasets:
        for seed in seeds:
            print(f"\n{ds.upper()} (split_seed={seed})")
            try:
                n_tr, n_va, n_te = preprocess(ds, seed)
                print(f"  Train={n_tr}, Valid={n_va}, Test={n_te}")
            except FileNotFoundError as e:
                print(f"  Skipped: {e}")
    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
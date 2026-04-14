#!/usr/bin/env python
"""
Preprocess Data Script

Loads raw CSV files, applies random scaffold split, and saves processed splits.
Reads split_seed from configs/base.yaml.

Usage:
    # Set split_seed in configs/base.yaml, then:
    python scripts/preprocess_data.py --dataset esol
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import prepare_dataset, save_splits, load_config


DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def preprocess_single_dataset(dataset_name: str, config_dir: str = "configs"):
    """Preprocess a single dataset using split_seed from config."""
    base_config_path = os.path.join(config_dir, "base.yaml")
    dataset_config_dir = os.path.join(config_dir, "datasets")

    # Read split_seed from base.yaml
    base_config = load_config(base_config_path)
    split_seed = base_config['data'].get('split_seed', 0)

    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name.upper()} (split_seed={split_seed})")
    print(f"{'='*50}")

    # Load and split data with split_seed
    train_df, valid_df, test_df, config = prepare_dataset(
        dataset_name,
        base_config_path=base_config_path,
        dataset_config_dir=dataset_config_dir,
        random_seed=split_seed
    )

    # Save splits into seed-specific subdirectory
    processed_dir = config['data']['processed_dir']
    seed_dir = os.path.join(processed_dir, f"seed_{split_seed}")
    os.makedirs(seed_dir, exist_ok=True)

    train_path = os.path.join(seed_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(seed_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(seed_dir, f"{dataset_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved splits to {seed_dir}/")

    # Also save to root processed_dir for backward compatibility
    save_splits(train_df, valid_df, test_df, processed_dir, dataset_name)

    return {
        'train_size': len(train_df),
        'valid_size': len(valid_df),
        'test_size': len(test_df),
        'split_seed': split_seed
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess molecular datasets")
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all'] + DATASETS,
        help='Dataset to process (default: all)'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs',
        help='Configuration directory'
    )
    args = parser.parse_args()

    # Change to project directory
    os.chdir(project_root)

    if args.dataset == 'all':
        datasets_to_process = DATASETS
    else:
        datasets_to_process = [args.dataset]

    # Read split_seed for display
    base_config = load_config(os.path.join(args.config_dir, "base.yaml"))
    split_seed = base_config['data'].get('split_seed', 0)

    print("\n" + "="*60)
    print("CONAN Project - Data Preprocessing")
    print(f"split_seed: {split_seed} (from configs/base.yaml)")
    print("="*60)

    results = {}
    for dataset in datasets_to_process:
        try:
            results[dataset] = preprocess_single_dataset(dataset, args.config_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {dataset} - please upload the raw data file first.")
            continue

    # Summary
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    for dataset, stats in results.items():
        print(f"{dataset} (split_seed={stats['split_seed']}): "
              f"Train={stats['train_size']}, Valid={stats['valid_size']}, Test={stats['test_size']}")


if __name__ == "__main__":
    main()
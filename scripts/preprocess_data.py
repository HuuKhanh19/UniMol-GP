#!/usr/bin/env python
"""
Preprocess Data Script

Loads raw CSV files, applies scaffold split, and saves processed splits.
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
    """Preprocess a single dataset."""
    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    base_config_path = os.path.join(config_dir, "base.yaml")
    dataset_config_dir = os.path.join(config_dir, "datasets")
    
    # Load and split data
    train_df, valid_df, test_df, config = prepare_dataset(
        dataset_name,
        base_config_path=base_config_path,
        dataset_config_dir=dataset_config_dir
    )
    
    # Save splits
    processed_dir = config['data']['processed_dir']
    save_splits(train_df, valid_df, test_df, processed_dir, dataset_name)
    
    return {
        'train_size': len(train_df),
        'valid_size': len(valid_df),
        'test_size': len(test_df)
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
    
    print("\n" + "="*60)
    print("CONAN Project - Data Preprocessing")
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
        print(f"{dataset}: Train={stats['train_size']}, Valid={stats['valid_size']}, Test={stats['test_size']}")


if __name__ == "__main__":
    main()

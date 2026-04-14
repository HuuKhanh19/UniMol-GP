#!/usr/bin/env python
"""
Step 1: Baseline UniMol Training

Train UniMol with standard gradient descent on molecular property prediction tasks.
This establishes the baseline performance for comparison with Steps 2 and 3.

Reads from configs/base.yaml:
  - split_seed: which scaffold split to use (change to 0,1,2,3,4)
  - random_seed: training seed (fixed at 42)

Usage:
    # Set split_seed in configs/base.yaml, then:
    python scripts/run_step1.py --dataset esol
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from src.data import load_config
from src.models import Step1Trainer
from src.utils import Timer, print_banner


DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def run_single_dataset(dataset_name: str, config_dir: str = "configs") -> dict:
    """
    Run Step 1 training on a single dataset.

    Reads split_seed from base.yaml to load the correct data split.
    Uses random_seed for training (model init, etc.).

    Args:
        dataset_name: Name of the dataset
        config_dir: Configuration directory

    Returns:
        Results dictionary
    """
    # Load configs
    base_config_path = os.path.join(config_dir, "base.yaml")
    dataset_config_path = os.path.join(config_dir, "datasets", f"{dataset_name}.yaml")

    base_config = load_config(base_config_path)
    dataset_config = load_config(dataset_config_path)

    # Merge configs
    config = {**base_config, **dataset_config}

    # Read seeds
    split_seed = config['data'].get('split_seed', 0)
    train_seed = config['data'].get('random_seed', 42)

    # Load data from seed-specific directory
    processed_dir = config['data']['processed_dir']
    seed_dir = os.path.join(processed_dir, f"seed_{split_seed}")

    train_path = os.path.join(seed_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(seed_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(seed_dir, f"{dataset_name}_test.csv")

    # Fallback to root processed_dir if seed dir not found
    if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        train_path = os.path.join(processed_dir, f"{dataset_name}_train.csv")
        valid_path = os.path.join(processed_dir, f"{dataset_name}_valid.csv")
        test_path = os.path.join(processed_dir, f"{dataset_name}_test.csv")

        if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
            print(f"Error: Preprocessed data not found for {dataset_name} (split_seed={split_seed})")
            print(f"Please run: python scripts/preprocess_data.py --dataset {dataset_name}")
            return None
        else:
            print(f"Warning: seed-specific split not found, using default from {processed_dir}")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded data (split_seed={split_seed}, train_seed={train_seed}) "
          f"- Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    smiles_col = 'smiles'
    target_col = 'target'

    # Output dir includes split_seed so different splits don't overwrite
    experiment_name = f"step1_baseline/seed_{split_seed}"

    # Create trainer — random_seed controls training reproducibility
    trainer = Step1Trainer(
        config=config,
        experiment_name=experiment_name
    )

    with Timer(f"Training {dataset_name} (split_seed={split_seed}, train_seed={train_seed})"):
        results = trainer.run(
            train_df,
            valid_df,
            test_df,
            smiles_column=smiles_col,
            target_column=target_col
        )

    if results:
        results['split_seed'] = split_seed
        results['train_seed'] = train_seed

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Baseline UniMol Training"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all'] + DATASETS,
        help='Dataset to train on (default: all)'
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

    # Read seeds for display
    base_config = load_config(os.path.join(args.config_dir, "base.yaml"))
    split_seed = base_config['data'].get('split_seed', 0)
    train_seed = base_config['data'].get('random_seed', 42)

    print_banner("CONAN Project - Step 1: Baseline Training")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"split_seed: {split_seed}  (scaffold split)")
    print(f"train_seed: {train_seed}  (model training)")

    if args.dataset == 'all':
        datasets = DATASETS
    else:
        datasets = [args.dataset]

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}\n")

        try:
            results = run_single_dataset(dataset, args.config_dir)
            if results:
                all_results[dataset] = results
        except Exception as e:
            print(f"Error training {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final summary
    print_banner("Final Summary - Step 1 Baseline Results")

    print("\n" + "-"*75)
    print(f"{'Dataset':<12} {'Task':<14} {'Metric':<8} {'Test Score':<12} "
          f"{'Split Seed':<12} {'Train Seed':<12}")
    print("-"*75)

    for dataset, results in all_results.items():
        task = results['task_type']
        metric = results['metric']
        s_seed = results.get('split_seed', 'N/A')
        t_seed = results.get('train_seed', 'N/A')
        test_score = results['test'].get(metric, 'N/A')
        if isinstance(test_score, float):
            print(f"{dataset:<12} {task:<14} {metric:<8} {test_score:<12.4f} "
                  f"{s_seed:<12} {t_seed:<12}")
        else:
            print(f"{dataset:<12} {task:<14} {metric:<8} {test_score:<12} "
                  f"{s_seed:<12} {t_seed:<12}")

    print("-"*75)

    # Save results
    output_dir = os.path.join("experiments", "step1_baseline", f"seed_{split_seed}")
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {summary_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
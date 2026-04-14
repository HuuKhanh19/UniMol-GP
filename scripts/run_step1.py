#!/usr/bin/env python
"""
Step 1: Baseline UniMol Training

Train UniMol with standard gradient descent on molecular property prediction tasks.
This establishes the baseline performance for comparison with Steps 2 and 3.
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
    
    # Load processed data
    processed_dir = config['data']['processed_dir']
    
    train_path = os.path.join(processed_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(processed_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(processed_dir, f"{dataset_name}_test.csv")
    
    # Check if preprocessed data exists
    if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        print(f"Error: Preprocessed data not found for {dataset_name}")
        print("Please run: python scripts/preprocess_data.py first")
        return None
    
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded data - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # After preprocessing, columns are standardized to 'smiles' and 'target'
    smiles_col = 'smiles'
    target_col = 'target'
    
    # Create trainer and run
    trainer = Step1Trainer(
        config=config,
        experiment_name="step1_baseline"
    )
    
    with Timer(f"Training {dataset_name}"):
        results = trainer.run(
            train_df,
            valid_df,
            test_df,
            smiles_column=smiles_col,
            target_column=target_col
        )
    
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
    
    print_banner("CONAN Project - Step 1: Baseline Training")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    print("\n" + "-"*60)
    print(f"{'Dataset':<15} {'Task':<15} {'Metric':<10} {'Test Score':<15}")
    print("-"*60)
    
    for dataset, results in all_results.items():
        task = results['task_type']
        metric = results['metric']
        test_score = results['test'].get(metric, 'N/A')
        if isinstance(test_score, float):
            print(f"{dataset:<15} {task:<15} {metric:<10} {test_score:<15.4f}")
        else:
            print(f"{dataset:<15} {task:<15} {metric:<10} {test_score:<15}")
    
    print("-"*60)
    
    # Save overall results
    output_dir = os.path.join("experiments", "step1_baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {summary_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 2: Train UniMol with EGGROLL Evolution Strategies

This script replaces gradient descent with EGGROLL for training UniMol models.
UPDATED: Uses full-batch fitness evaluation (no mini-batch option).

Usage:
    python scripts/run_step2.py --dataset esol
    python scripts/run_step2.py --dataset all
    python scripts/run_step2.py --dataset esol --population_size 32 --rank 16
"""

import os
import sys
import argparse
import json
import yaml
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_config(dataset_name: str) -> dict:
    """Load and merge base config with dataset-specific config."""
    base_path = os.path.join(project_root, 'configs', 'base.yaml')
    dataset_path = os.path.join(project_root, 'configs', 'datasets', f'{dataset_name}.yaml')
    
    with open(base_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Merge dataset config
        for key, value in dataset_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return config


def run_step2(
    dataset_name: str,
    population_size: int = None,
    rank: int = None,
    sigma: float = None,
    learning_rate: float = None,
    num_generations: int = None,
    eval_chunk_size: int = None,
    patience: int = None,
    weight_decay: float = None,
    rank_transform: bool = None,
    lr_decay: float = None,
    sigma_decay: float = None
):
    """Run Step 2 EGGROLL training for a dataset."""
    from src.data.loader import DatasetLoader
    from src.trainers.step2_eggroll import Step2Trainer
    
    print(f"\n{'='*60}")
    print(f"Step 2: EGGROLL Training (Full-Batch) for {dataset_name}")
    print(f"Start time: {datetime.now()}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(dataset_name)
    
    # Override EGGROLL parameters if provided
    if population_size is not None:
        config['eggroll']['population_size'] = population_size
    if rank is not None:
        config['eggroll']['rank'] = rank
    if sigma is not None:
        config['eggroll']['sigma'] = sigma
    if learning_rate is not None:
        config['eggroll']['learning_rate'] = learning_rate
    if num_generations is not None:
        config['eggroll']['num_generations'] = num_generations
    if eval_chunk_size is not None:
        config['eggroll']['eval_chunk_size'] = eval_chunk_size
    if patience is not None:
        config['eggroll']['patience'] = patience
    if weight_decay is not None:
        config['eggroll']['weight_decay'] = weight_decay
    if rank_transform is not None:
        config['eggroll']['rank_transform'] = rank_transform
    if lr_decay is not None:
        config['eggroll']['lr_decay'] = lr_decay
    if sigma_decay is not None:
        config['eggroll']['sigma_decay'] = sigma_decay
    
    # Print EGGROLL config
    print("\nEGGROLL Configuration (Full-Batch Mode):")
    for key, value in config['eggroll'].items():
        print(f"  {key}: {value}")
    
    # Load data
    loader = DatasetLoader(config)
    train_data, valid_data, test_data = loader.load()
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_data)} samples (full-batch)")
    print(f"  Valid: {len(valid_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Create trainer
    trainer = Step2Trainer(config, experiment_name="step2_eggroll")
    
    # Run training
    results = trainer.run(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        smiles_column='smiles',  # Standardized by DatasetLoader
        target_column='target'   # Standardized by DatasetLoader
    )
    
    print(f"\nTraining {dataset_name} completed")
    print(f"End time: {datetime.now()}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Train UniMol with EGGROLL Evolution Strategies (Full-Batch)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='esol',
        choices=['esol', 'freesolv', 'lipo', 'bace', 'all'],
        help='Dataset to train on (default: esol)'
    )
    parser.add_argument(
        '--population_size', '-N',
        type=int,
        default=None,
        help='Population size (number of perturbations, default: 32)'
    )
    parser.add_argument(
        '--rank', '-r',
        type=int,
        default=None,
        help='Rank of low-rank perturbations (default: 16)'
    )
    parser.add_argument(
        '--sigma', '-s',
        type=float,
        default=None,
        help='Noise scale for perturbations (default: 0.01)'
    )
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=None,
        help='Learning rate (step size, default: 0.1)'
    )
    parser.add_argument(
        '--num_generations', '-g',
        type=int,
        default=None,
        help='Number of evolution generations (default: 400)'
    )
    parser.add_argument(
        '--eval_chunk_size', '-c',
        type=int,
        default=None,
        help='Chunk size for forward pass to avoid OOM (default: 64)'
    )
    parser.add_argument(
        '--patience', '-p',
        type=int,
        default=None,
        help='Early stopping patience (default: 200)'
    )
    parser.add_argument(
        '--weight_decay', '-wd',
        type=float,
        default=None,
        help='Weight decay for regularization (default: 0.0)'
    )
    parser.add_argument(
        '--rank_transform',
        action='store_true',
        help='Use rank-based fitness shaping (default: True)'
    )
    parser.add_argument(
        '--no_rank_transform',
        action='store_true',
        help='Disable rank-based fitness shaping'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=None,
        help='Learning rate decay per generation (default: 0.99)'
    )
    parser.add_argument(
        '--sigma_decay',
        type=float,
        default=None,
        help='Sigma decay per generation (default: 0.99)'
    )
    
    args = parser.parse_args()
    
    # Handle rank_transform flag
    rank_transform = None
    if args.rank_transform:
        rank_transform = True
    elif args.no_rank_transform:
        rank_transform = False
    
    # Datasets to run
    if args.dataset == 'all':
        datasets = ['esol', 'freesolv', 'lipo', 'bace']
    else:
        datasets = [args.dataset]
    
    # Run for each dataset
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_step2(
                dataset,
                population_size=args.population_size,
                rank=args.rank,
                sigma=args.sigma,
                learning_rate=args.learning_rate,
                num_generations=args.num_generations,
                eval_chunk_size=args.eval_chunk_size,
                patience=args.patience,
                weight_decay=args.weight_decay,
                rank_transform=rank_transform,
                lr_decay=args.lr_decay,
                sigma_decay=args.sigma_decay
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"\nError training {dataset}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"         Step 2 Summary - EGGROLL Results (Full-Batch)          ")
    print(f"{'='*60}\n")
    
    print("-" * 60)
    print(f"{'Dataset':<15} {'Task':<15} {'Metric':<10} {'Test Score':<12}")
    print("-" * 60)
    
    for dataset, result in all_results.items():
        if 'error' in result:
            print(f"{dataset:<15} ERROR: {result['error'][:30]}")
        else:
            task = result.get('task_type', 'unknown')
            metric = result.get('metric', 'unknown')
            test_score = result.get('test', {}).get(metric, 0.0)
            print(f"{dataset:<15} {task:<15} {metric:<10} {test_score:<12.4f}")
    
    print("-" * 60)
    
    # Save summary only when running multiple datasets
    if len(datasets) > 1:
        summary_path = os.path.join(project_root, 'experiments', 'step2_eggroll', 'summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        # Convert for JSON serialization
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        all_results = convert_numpy(all_results)
        
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
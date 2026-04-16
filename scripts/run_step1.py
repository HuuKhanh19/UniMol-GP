#!/usr/bin/env python
"""
Step 1: Baseline UniMol Training (Gradient Descent).

Priority: CLI > config.yaml > DEFAULTS below.

Usage:
    python scripts/run_step1.py --dataset esol
    python scripts/run_step1.py --dataset esol --split-seed 2 --epochs 50
    python scripts/run_step1.py --dataset esol --gpu-id 1
"""

import os, sys, argparse, json, yaml, logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from src.data import DATASET_NAMES, get_dataset_info
from src.data.datasets import PROCESSED_DIR, OUTPUT_DIR
from src.models import Step1Trainer
from src.utils import Timer, print_banner

# ── ALL defaults ─────────────────────────────────────────────────────────

DEFAULTS = {
    # Shared
    'split_seed':       0,
    'random_seed':      42,
    'n_confomer':       1,
    'gpu_id':           0,
    # Step 1 tunable (config.yaml)
    'epochs':           100,
    'batch_size':       32,
    'learning_rate':    0.0001,
    'patience':         10,
    # Rarely changed
    'warmup_ratio':     0.03,
    'max_norm':         5.0,
    'target_normalize': 'auto',
    'remove_hs':        True,
    'use_gpu':          True,
    'use_amp':          True,
    'model_name':       'unimolv1',
    'freeze_layers':    None,
}

CONFIG_KEYS = {'split_seed', 'n_confomer', 'gpu_id',
               'epochs', 'batch_size', 'learning_rate', 'patience'}


def load_config(path='config.yaml'):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_params(args, cfg):
    params = {}
    for key, default in DEFAULTS.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            params[key] = cli_val
        elif key in CONFIG_KEYS and key in cfg:
            params[key] = cfg[key]
        else:
            params[key] = default
    return params


def load_split(dataset_name, split_seed):
    seed_dir = os.path.join(PROCESSED_DIR, dataset_name, f"seed_{split_seed}")
    paths = {s: os.path.join(seed_dir, f"{dataset_name}_{s}.csv")
             for s in ('train', 'valid', 'test')}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Data not found at {seed_dir}/\n"
            f"Run: python scripts/preprocess_data.py "
            f"--dataset {dataset_name} --split-seed {split_seed}")
    return tuple(pd.read_csv(paths[s]) for s in ('train', 'valid', 'test'))


def set_clean_log_format():
    fmt = logging.Formatter('%(message)s')
    for name in ['Uni-Mol Tools', 'unimol', '']:
        lg = logging.getLogger(name)
        for h in lg.handlers:
            h.setFormatter(fmt)


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Baseline UniMol Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES)
    # Shared
    parser.add_argument('--split-seed',    type=int,   default=None)
    parser.add_argument('--n-confomer',    type=int,   default=None)
    parser.add_argument('--gpu-id',        type=int,   default=None)
    # Step 1 tunable
    parser.add_argument('--epochs',        type=int,   default=None)
    parser.add_argument('--batch-size',    type=int,   default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--patience',      type=int,   default=None)
    # Rarely changed
    parser.add_argument('--random-seed',      type=int,   default=None)
    parser.add_argument('--warmup-ratio',     type=float, default=None)
    parser.add_argument('--max-norm',         type=float, default=None)
    parser.add_argument('--target-normalize', type=str,   default=None)
    parser.add_argument('--no-remove-hs',     action='store_true')
    parser.add_argument('--no-gpu',           action='store_true')
    parser.add_argument('--no-amp',           action='store_true')
    parser.add_argument('--model-name',       type=str,   default=None)
    parser.add_argument('--freeze-layers',    type=str,   default=None)
    # Experiment
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--config',  type=str, default='config.yaml')

    args = parser.parse_args()
    args.remove_hs = False if args.no_remove_hs else None
    args.use_gpu   = False if args.no_gpu else None
    args.use_amp   = False if args.no_amp else None

    os.chdir(project_root)
    cfg = load_config(args.config)
    params = resolve_params(args, cfg)

    dataset_info = get_dataset_info(args.dataset)
    split_seed = params['split_seed']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Header
    print_banner("UniMol-GP -- Step 1: Baseline Training")
    print(f"Time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Dataset     : {args.dataset} ({dataset_info['task_type']}, {dataset_info['metric']})")
    print(f"split_seed  : {split_seed}")
    print(f"random_seed : {params['random_seed']}")
    print(f"n_confomer  : {params['n_confomer']}")
    print(f"gpu_id      : {params['gpu_id']}")
    print(f"epochs      : {params['epochs']}")
    print(f"batch_size  : {params['batch_size']}")
    print(f"lr          : {params['learning_rate']}")
    print(f"patience    : {params['patience']}")
    print(f"warmup      : {params['warmup_ratio']}")
    print(f"max_norm    : {params['max_norm']}")
    print(f"normalize   : {params['target_normalize']}")
    print(f"remove_hs   : {params['remove_hs']}")
    print(f"GPU/AMP     : {params['use_gpu']}/{params['use_amp']}")
    if not args.no_save:
        print(f"Save to     : {OUTPUT_DIR}/step1/{args.dataset}/seed_{split_seed}/{timestamp}/")

    # Load data
    train_df, valid_df, test_df = load_split(args.dataset, split_seed)
    print(f"\nData -- Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Clean logs
    set_clean_log_format()

    # Experiment path: experiments/step1/{dataset}/seed_{X}/{timestamp}
    experiment_name = f"step1/{args.dataset}/seed_{split_seed}/{timestamp}"
    trainer = Step1Trainer(
        params=params, dataset_info=dataset_info, experiment_name=experiment_name,
    )

    with Timer(f"Training {args.dataset} (split_seed={split_seed})"):
        results = trainer.run(train_df, valid_df, test_df)

    if results is None:
        sys.exit(1)

    results['split_seed'] = split_seed
    results['train_seed'] = params['random_seed']
    results['timestamp'] = timestamp
    results['params'] = params

    if not args.no_save:
        out_dir = os.path.join(OUTPUT_DIR, 'step1', args.dataset,
                               f"seed_{split_seed}", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved -- {out_dir}/results.json")
    else:
        print("(--no-save: results not saved)")

    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
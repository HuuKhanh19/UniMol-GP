#!/usr/bin/env python
"""
Step 2: Train UniMol end-to-end with EGGROLL Evolution Strategies.

Priority: CLI > config.yaml > DEFAULTS below.

Usage:
    python scripts/run_step2.py --dataset esol
    python scripts/run_step2.py --dataset esol --split-seed 2 --sigma 0.005
    python scripts/run_step2.py --dataset esol --gpu-id 1
"""

import os, sys, argparse, json, yaml, logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from src.data import DATASET_NAMES, get_dataset_info
from src.data.datasets import PROCESSED_DIR, OUTPUT_DIR
from src.trainers.step2_eggroll import Step2Trainer
from src.utils import Timer, print_banner

# ── ALL defaults ─────────────────────────────────────────────────────────

DEFAULTS = {
    # Shared
    'split_seed':         0,
    'random_seed':        42,
    'gpu_id':             0,
    'use_gpu':            True,
    # EGGROLL core (config.yaml eggroll:)
    'population_size':    32,
    'rank':               16,
    'sigma':              0.01,
    'eggroll_lr':         0.1,
    'num_generations':    400,
    'eggroll_patience':   200,
    'eval_chunk_size':    64,
    'lr_decay':           0.99,
    'sigma_decay':        0.99,
    # EGGROLL advanced
    'use_antithetic':     True,
    'normalize_fitness':  True,
    'rank_transform':     True,
    'centered_rank':      True,
    'weight_decay':       0.0,
}


def load_config(path='config.yaml'):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _pick(cli, cfg, default):
    if cli is not None: return cli
    if cfg is not None: return cfg
    return default


def resolve_params(args, cfg):
    ecfg = cfg.get('eggroll', {})
    params = {}
    # Shared
    params['split_seed'] = _pick(args.split_seed, cfg.get('split_seed'), DEFAULTS['split_seed'])
    params['random_seed'] = _pick(args.random_seed, None, DEFAULTS['random_seed'])
    params['gpu_id'] = _pick(args.gpu_id, cfg.get('gpu_id'), DEFAULTS['gpu_id'])
    params['use_gpu'] = DEFAULTS['use_gpu'] if not args.no_gpu else False
    # EGGROLL
    params['population_size'] = _pick(args.population_size, ecfg.get('population_size'), DEFAULTS['population_size'])
    params['rank'] = _pick(args.rank, ecfg.get('rank'), DEFAULTS['rank'])
    params['sigma'] = _pick(args.sigma, ecfg.get('sigma'), DEFAULTS['sigma'])
    params['eggroll_lr'] = _pick(args.eggroll_lr, ecfg.get('learning_rate'), DEFAULTS['eggroll_lr'])
    params['num_generations'] = _pick(args.num_generations, ecfg.get('num_generations'), DEFAULTS['num_generations'])
    params['eggroll_patience'] = _pick(args.eggroll_patience, ecfg.get('patience'), DEFAULTS['eggroll_patience'])
    params['eval_chunk_size'] = _pick(args.eval_chunk_size, ecfg.get('eval_chunk_size'), DEFAULTS['eval_chunk_size'])
    params['lr_decay'] = _pick(args.lr_decay, ecfg.get('lr_decay'), DEFAULTS['lr_decay'])
    params['sigma_decay'] = _pick(args.sigma_decay, ecfg.get('sigma_decay'), DEFAULTS['sigma_decay'])
    for k in ('use_antithetic', 'normalize_fitness', 'rank_transform', 'centered_rank', 'weight_decay'):
        params[k] = DEFAULTS[k]
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
        description="Step 2: UniMol + EGGROLL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES)
    # Shared
    parser.add_argument('--split-seed',      type=int,   default=None)
    parser.add_argument('--random-seed',     type=int,   default=None)
    parser.add_argument('--gpu-id',          type=int,   default=None)
    parser.add_argument('--no-gpu',          action='store_true')
    # EGGROLL
    parser.add_argument('--population-size', type=int,   default=None)
    parser.add_argument('--rank',            type=int,   default=None)
    parser.add_argument('--sigma',           type=float, default=None)
    parser.add_argument('--eggroll-lr',      type=float, default=None)
    parser.add_argument('--num-generations', type=int,   default=None)
    parser.add_argument('--eggroll-patience',type=int,   default=None)
    parser.add_argument('--eval-chunk-size', type=int,   default=None)
    parser.add_argument('--lr-decay',        type=float, default=None)
    parser.add_argument('--sigma-decay',     type=float, default=None)
    # Experiment
    parser.add_argument('--no-save',         action='store_true')
    parser.add_argument('--config',          type=str,   default='config.yaml')

    args = parser.parse_args()
    os.chdir(project_root)
    cfg = load_config(args.config)

    # Check n_confomer
    n_conf = cfg.get('n_confomer', 1)
    if n_conf > 1:
        print(f"ERROR: Step 2 (EGGROLL) only supports n_confomer=1, "
              f"but config.yaml has n_confomer={n_conf}.")
        sys.exit(1)

    params = resolve_params(args, cfg)
    dataset_info = get_dataset_info(args.dataset)
    split_seed = params['split_seed']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Header
    print_banner("UniMol-GP -- Step 2: EGGROLL Training")
    print(f"Time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Dataset     : {args.dataset} ({dataset_info['task_type']}, {dataset_info['metric']})")
    print(f"split_seed  : {split_seed}")
    print(f"random_seed : {params['random_seed']}")
    print(f"gpu_id      : {params['gpu_id']}")
    print(f"--- EGGROLL ---")
    print(f"N (pop)     : {params['population_size']}")
    print(f"r (rank)    : {params['rank']}")
    print(f"sigma       : {params['sigma']}")
    print(f"lr          : {params['eggroll_lr']}")
    print(f"Generations : {params['num_generations']}")
    print(f"Patience    : {params['eggroll_patience']}")
    print(f"Chunk size  : {params['eval_chunk_size']}")
    print(f"LR decay    : {params['lr_decay']}")
    print(f"Sigma decay : {params['sigma_decay']}")
    if not args.no_save:
        print(f"Save to     : {OUTPUT_DIR}/step2/{args.dataset}/seed_{split_seed}/{timestamp}/")

    # Load data
    train_df, valid_df, test_df = load_split(args.dataset, split_seed)
    print(f"\nData -- Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    set_clean_log_format()

    # Experiment path: experiments/step2/{dataset}/seed_{X}/{timestamp}
    experiment_name = f"step2/{args.dataset}/seed_{split_seed}/{timestamp}"
    trainer = Step2Trainer(
        params=params, dataset_info=dataset_info, experiment_name=experiment_name,
    )

    with Timer(f"EGGROLL training {args.dataset} (split_seed={split_seed})"):
        results = trainer.run(train_df, valid_df, test_df)

    if results is None:
        sys.exit(1)

    results['split_seed'] = split_seed
    results['random_seed'] = params['random_seed']
    results['timestamp'] = timestamp
    results['params'] = params

    if not args.no_save:
        out_dir = os.path.join(OUTPUT_DIR, 'step2', args.dataset,
                               f"seed_{split_seed}", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved -- {out_dir}/results.json")

    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
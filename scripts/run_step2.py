#!/usr/bin/env python
"""
Step 2: LoRA-subspace adaptation of UniMol-v1 on a frozen pretrained backbone.

    --substep 2.1   ->  Gradient Descent on LoRA + head   (reference)
    --substep 2.2   ->  EGGROLL / Evolution Strategies on LoRA + head

Mirrors scripts/run_step1.py (same data split, same target_scaler, same RMSE eval).
Target: 5-seed mean test-RMSE <= 0.8523  (= 0.8023 baseline + 0.05).

Usage:
    python scripts/run_step2.py --dataset esol --substep 2.1 --split-seed 0
    python scripts/run_step2.py --dataset esol --substep 2.2 --split-seed 0 \
        --lora-rank 8 --sigma 0.01 --es-lr 1e-3 --popsize 256 --es-steps 600
"""

import os, sys, argparse, json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from src.data import DATASET_NAMES, get_dataset_info
from src.data.datasets import PROCESSED_DIR, OUTPUT_DIR
from src.models.es_lora_trainer import Step2Trainer
from src.utils import Timer, print_banner

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULTS = {
    # shared
    'split_seed':       0,
    'random_seed':      42,
    'n_confomer':       1,
    'gpu_id':           0,
    # frozen-backbone training (GD path reuses these; ES uses patience only)
    'epochs':           100,
    'batch_size':       32,
    'learning_rate':    1e-4,      # Step 2.1: Adam on LoRA+head (mirrors baseline)
    'patience':         10,        # GD (2.1) early-stop patience in EPOCHS (ES uses es_patience)
    'warmup_ratio':     0.03,
    'max_norm':         5.0,
    'target_normalize': 'auto',
    'remove_hs':        True,
    'use_gpu':          True,
    'use_amp':          True,
    'model_name':       'unimolv1',
    # Step 2 -- LoRA
    'method':           'gd',      # 'gd' (2.1) | 'es' (2.2)
    'lora_rank':        16,
    'lora_alpha':       16.0,
    # Step 2.2 -- ES / EGGROLL  (defaults tuned to fit ~1h/seed on 1x RTX 5070 Ti)
    'es_sigma':         0.001,      # search range [1e-3, 1e-2]; smaller -> closer to gradient
    'es_lr':            0.005,      # AdamW outer LR
    'es_lr_decay':      0.05,       # cosine final-LR fraction; 1.0 = constant (off). Try 0.0-0.05
    'es_popsize':       32,       # members/step (antithetic). 128-512 typical
    'es_steps':         400,      # ES updates. budget ~= es_steps * popsize forwards
    'es_log_every':     10,        # PRINT cadence only -- val mse is computed EVERY step
    'es_patience':      200,       # early-stop patience IN STEPS (no val improvement)
    'es_weight_decay':  0.0,
    'es_rank_transform': True,     # centered-rank fitness shaping (robust to RMSE outliers)
    'es_data_batch':    0,       # ES fitness minibatch; 0 = full-batch (cleaner but ~7x slower)
}

SUBSTEP_TO_METHOD = {'2.1': 'gd', '2.2': 'es'}


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


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: LoRA-subspace adaptation (GD / EGGROLL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES)
    parser.add_argument('--substep', type=str, default='2.1', choices=['2.1', '2.2'],
                        help="2.1 = GD on LoRA, 2.2 = EGGROLL on LoRA")
    # shared
    parser.add_argument('--split-seed',    type=int,   default=None)
    parser.add_argument('--random-seed',   type=int,   default=None)
    parser.add_argument('--n-confomer',    type=int,   default=None)
    parser.add_argument('--gpu-id',        type=int,   default=None)
    parser.add_argument('--epochs',        type=int,   default=None)
    parser.add_argument('--batch-size',    type=int,   default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--patience',      type=int,   default=None)
    parser.add_argument('--no-amp',        action='store_true')
    parser.add_argument('--no-gpu',        action='store_true')
    # LoRA
    parser.add_argument('--lora-rank',  type=int,   default=None)
    parser.add_argument('--lora-alpha', type=float, default=None)
    # ES / EGGROLL
    parser.add_argument('--sigma',          type=float, default=None, dest='es_sigma')
    parser.add_argument('--es-lr',          type=float, default=None)
    parser.add_argument('--es-lr-decay',    type=float, default=None,
                        help="cosine final-LR fraction (1.0=off, 0.0=decay to zero, 0.05=to 5%%)")
    parser.add_argument('--popsize',        type=int,   default=None, dest='es_popsize')
    parser.add_argument('--es-steps',       type=int,   default=None)
    parser.add_argument('--es-log-every',   type=int,   default=None,
                        help="print cadence only; val mse is computed every step")
    parser.add_argument('--es-patience',    type=int,   default=None,
                        help="early-stop patience in STEPS (no val improvement)")
    parser.add_argument('--es-weight-decay', type=float, default=None)
    parser.add_argument('--es-data-batch',  type=int,   default=None,
                        help="ES fitness minibatch size; 0 = full-batch")
    parser.add_argument('--no-rank-transform', action='store_true')
    # experiment
    parser.add_argument('--no-save', action='store_true')

    args = parser.parse_args()

    params = dict(DEFAULTS)
    params['method'] = SUBSTEP_TO_METHOD[args.substep]
    for key in list(DEFAULTS.keys()):
        v = getattr(args, key, None)
        if v is not None:
            params[key] = v
    if args.no_amp:  params['use_amp'] = False
    if args.no_gpu:  params['use_gpu'] = False
    if args.no_rank_transform: params['es_rank_transform'] = False

    os.chdir(project_root)
    dataset_info = get_dataset_info(args.dataset)
    split_seed = params['split_seed']
    method = params['method']
    substep_tag = f"{args.substep}_{method}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print_banner(f"UniMol-GP -- Step {args.substep}: "
                 f"{'GD on LoRA' if method == 'gd' else 'EGGROLL on LoRA'}")
    print(f"Time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Dataset     : {args.dataset} ({dataset_info['task_type']}, {dataset_info['metric']})")
    print(f"substep     : {args.substep}  (method={method})")
    print(f"split_seed  : {split_seed}")
    print(f"gpu_id      : {params['gpu_id']}")
    print(f"LoRA        : rank={params['lora_rank']}, alpha={params['lora_alpha']}")
    if method == 'gd':
        print(f"GD          : epochs={params['epochs']}, lr={params['learning_rate']}, "
              f"batch={params['batch_size']}, patience={params['patience']}")
    else:
        print(f"ES          : sigma={params['es_sigma']}, es_lr={params['es_lr']}, "
              f"lr_decay={params['es_lr_decay']}{' (cosine)' if params['es_lr_decay'] < 1.0 else ' (constant)'}, "
              f"pop={params['es_popsize']}, steps={params['es_steps']}, "
              f"log_every={params['es_log_every']}, es_patience={params['es_patience']} steps, "
              f"data_batch={'full' if params['es_data_batch'] == 0 else params['es_data_batch']}, "
              f"rank_transform={params['es_rank_transform']}  (val computed EVERY step)")
    print(f"GPU/AMP     : {params['use_gpu']}/{params['use_amp']}")

    train_df, valid_df, test_df = load_split(args.dataset, split_seed)
    print(f"\nData -- Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    experiment_name = f"step2/{args.dataset}/{substep_tag}/seed_{split_seed}/{timestamp}"
    print(f"Save to     : {os.path.join(OUTPUT_DIR, experiment_name)}/\n")

    with Timer() as t:
        trainer = Step2Trainer(params, dataset_info, experiment_name)
        results = trainer.run(train_df, valid_df, test_df,
                              smiles_column='smiles', target_column='target')

    results.update({
        'split_seed': split_seed,
        'train_seed': params['random_seed'],
        'substep': args.substep,
        'timestamp': timestamp,
        'params': params,
    })
    print(f"Training {args.dataset} (split_seed={split_seed}, {substep_tag}) "
          f"completed in {t.elapsed:.2f} seconds")

    if not args.no_save:
        out_dir = os.path.join(OUTPUT_DIR, experiment_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved -- {os.path.join(out_dir, 'results.json')}")
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == '__main__':
    main()
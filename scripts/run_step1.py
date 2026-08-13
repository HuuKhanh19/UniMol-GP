#!/usr/bin/env python
"""
Step 1: UniMol v1 baseline training (gradient descent).

All configuration lives in argparse: ``add_argument`` is the single declaration
site for every knob and its default, and ``--help`` is the full reference. There
is no config file.

Usage:
    python scripts/run_step1.py --dataset esol
    python scripts/run_step1.py --dataset esol --split-seed 2 --epochs 50
    python scripts/run_step1.py --dataset esol --split random --split-seed 2
    python scripts/run_step1.py --dataset esol --gpu-id 1 --no-amp
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data import DATASET_NAMES, get_dataset_info  # noqa: E402
from src.data.datasets import (  # noqa: E402
    DEFAULT_SPLIT,
    OUTPUT_DIR,
    SPLIT_TYPES,
    experiment_name,
    split_dir,
)
from src.models import Step1Trainer  # noqa: E402
from src.utils import Timer, print_banner, save_json  # noqa: E402

#: This project targets UniMol v1 only, so the model is not a CLI knob.
MODEL_NAME = 'unimolv1'

#: Parsed arguments that steer the script rather than the model.
NON_PARAM_DESTS = frozenset({'dataset', 'no_save'})


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Declare every option, with its default, in one place."""
    parser = argparse.ArgumentParser(
        prog='run_step1.py',
        description='Step 1: UniMol v1 baseline training (gradient descent).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', required=True, choices=DATASET_NAMES,
                        default=argparse.SUPPRESS,
                        help='dataset key from the registry')

    split = parser.add_argument_group('data split')
    split.add_argument('--split', default=DEFAULT_SPLIT, choices=SPLIT_TYPES,
                       help='which split family to read; must match how '
                            'preprocess_data.py was run')
    split.add_argument('--split-seed', type=int, default=0,
                       help='which split to train on')
    split.add_argument('--random-seed', type=int, default=42,
                       help='training seed; unrelated to --split-seed')

    train = parser.add_argument_group('training')
    train.add_argument('--epochs', type=int, default=100,
                       help='maximum epochs; early stopping may cut it short')
    train.add_argument('--batch-size', type=int, default=32,
                       help='molecules per optimiser step')
    train.add_argument('--learning-rate', type=float, default=1e-4,
                       help='peak LR after warmup')
    train.add_argument('--patience', type=int, default=10,
                       help='early-stopping patience, in epochs')
    train.add_argument('--warmup-ratio', type=float, default=0.03,
                       help='fraction of total steps spent warming up the LR')
    train.add_argument('--max-norm', type=float, default=5.0,
                       help='gradient-clipping norm')

    feat = parser.add_argument_group('featurisation')
    feat.add_argument('--target-normalize', default='auto',
                      help="target scaler: 'auto' or 'none'")
    feat.add_argument('--remove-hs', action='store_true', default=False,
                      help='strip hydrogens and load the no-H checkpoint; the '
                           'default keeps every hydrogen (all_h), matching the '
                           'unimol_tools default')
    feat.add_argument('--freeze-layers', default=None,
                      help='comma-separated encoder layers to freeze')

    hardware = parser.add_argument_group('hardware')
    hardware.add_argument('--gpu-id', type=int, default=0,
                          help='CUDA device index; ignored without a GPU')
    hardware.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                          default=True,
                          help='force CPU (use_gpu default: %(default)s)')
    hardware.add_argument('--no-amp', dest='use_amp', action='store_false',
                          default=True,
                          help='disable mixed precision '
                               '(use_amp default: %(default)s)')

    output = parser.add_argument_group('output')
    output.add_argument('--no-save', action='store_true',
                        help='run without writing results.json')
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line. Defaults come from build_parser(), nowhere else."""
    return build_parser().parse_args(argv)


def resolve_unimol_source() -> str:
    """Return the unimol_tools package dir, refusing anything outside this repo.

    This repo ships a patched fork under unimol_source/. If the environment has
    a different copy installed -- a second checkout, or the upstream package
    from PyPI -- runs would silently use that instead of the code being edited
    here, so stop rather than produce results nobody can trace back.
    """
    import unimol_tools

    pkg_dir = os.path.dirname(os.path.abspath(unimol_tools.__file__))
    expected = os.path.join(PROJECT_ROOT, 'unimol_source')
    if not os.path.normcase(pkg_dir).startswith(
            os.path.normcase(expected) + os.sep):
        raise SystemExit(
            f'\nunimol_tools resolves to:\n    {pkg_dir}\n'
            f'but this repo ships its own patched fork at:\n    {expected}\n\n'
            f'Reinstall it so runs use the code in this checkout:\n'
            f'    pip uninstall -y unimol_tools\n'
            f'    pip install -e "{expected}"\n')
    return pkg_dir


def checkpoint_name(remove_hs: bool) -> str:
    """Pretrained file unimol_tools will load for this remove_hs setting."""
    from unimol_tools.config import MODEL_CONFIG

    key = 'molecule_no_h' if remove_hs else 'molecule_all_h'
    return MODEL_CONFIG['weight'][key]


def training_params(args: argparse.Namespace) -> dict[str, Any]:
    """Everything the trainer needs, keyed exactly as UniMolWrapper expects."""
    params = {k: v for k, v in vars(args).items() if k not in NON_PARAM_DESTS}
    params['model_name'] = MODEL_NAME
    return params


# ── Data ─────────────────────────────────────────────────────────────────

def load_split(dataset_name: str, split_seed: int, split: str = DEFAULT_SPLIT
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train/valid/test CSVs written by preprocess_data.py."""
    seed_dir = split_dir(dataset_name, split_seed, split)
    paths = {s: os.path.join(seed_dir, f'{dataset_name}_{s}.csv')
             for s in ('train', 'valid', 'test')}
    if any(not os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError(
            f'Data not found at {seed_dir}/\n'
            f'Run: python scripts/preprocess_data.py '
            f'--dataset {dataset_name} --split {split} '
            f'--split-seed {split_seed}')
    return tuple(pd.read_csv(paths[s]) for s in ('train', 'valid', 'test'))


# ── Console ──────────────────────────────────────────────────────────────

def set_clean_log_format() -> None:
    """Strip timestamps/levels from unimol_tools log lines."""
    fmt = logging.Formatter('%(message)s')
    for name in ('Uni-Mol Tools', 'unimol', ''):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(fmt)


def print_header(args: argparse.Namespace, params: dict[str, Any],
                 dataset_info: dict[str, Any], out_dir: str | None,
                 unimol_dir: str) -> None:
    print_banner('UniMol-GP -- Step 1: Baseline Training')
    rows: list[tuple[str, Any]] = [
        ('Time', f'{datetime.now():%Y-%m-%d %H:%M:%S}'),
        ('Dataset', (f"{args.dataset} "
                     f"({dataset_info['task_type']}, {dataset_info['metric']})")),
        ('Model', params['model_name']),
        ('unimol_tools', unimol_dir),
        ('Checkpoint', checkpoint_name(params['remove_hs'])),
        ('Split', f"{params['split']} (seed {params['split_seed']})"),
        ('Random seed', params['random_seed']),
        ('Epochs', params['epochs']),
        ('Batch size', params['batch_size']),
        ('Learning rate', params['learning_rate']),
        ('Patience', params['patience']),
        ('Warmup ratio', params['warmup_ratio']),
        ('Max norm', params['max_norm']),
        ('Target scaler', params['target_normalize']),
        ('Remove Hs', f"{params['remove_hs']} "
                      f"({'no_h' if params['remove_hs'] else 'all_h'})"),
        ('GPU / AMP', (f"{params['use_gpu']} / {params['use_amp']} "
                       f"(gpu_id={params['gpu_id']})")),
        ('Save to', out_dir or '(--no-save)'),
    ]
    for label, value in rows:
        print(f'{label:<14}: {value}')


# ── Entry point ──────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)

    # Fail before any training happens if the wrong unimol_tools is installed.
    unimol_dir = resolve_unimol_source()

    params = training_params(args)
    dataset_info = get_dataset_info(args.dataset)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = experiment_name('step1', args.dataset, params['split_seed'],
                               params['split'], timestamp)
    out_dir = None if args.no_save else os.path.join(OUTPUT_DIR, run_name)

    print_header(args, params, dataset_info, out_dir, unimol_dir)

    train_df, valid_df, test_df = load_split(
        args.dataset, params['split_seed'], params['split'])
    print(f'\nData -- Train: {len(train_df)}, '
          f'Valid: {len(valid_df)}, Test: {len(test_df)}')

    set_clean_log_format()
    trainer = Step1Trainer(params=params, dataset_info=dataset_info,
                           experiment_name=run_name)

    with Timer(f"Training {args.dataset} "
               f"({params['split']} split_seed={params['split_seed']})"):
        results = trainer.run(train_df, valid_df, test_df)

    results.update({
        'split': params['split'],
        'split_seed': params['split_seed'],
        'train_seed': params['random_seed'],
        'timestamp': timestamp,
        'params': params,
    })

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, 'results.json')
        save_json(results, results_path)
        print(f'Results saved -- {results_path}')
    else:
        print('(--no-save: results not saved)')

    print(f'End: {datetime.now():%Y-%m-%d %H:%M:%S}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

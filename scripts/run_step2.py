#!/usr/bin/env python
"""
Step 2: symbolic (GP) head + EGGROLL, co-adapted end to end.

Replaces UniMol's ``LinearHead`` with k formula trees over disjoint regions of
the 512-d CLS embedding, merged by ridge, and fine-tunes the last transformer
blocks with low-rank evolution strategies instead of gradient descent.

Like ``run_step1.py``, all configuration lives in argparse: ``add_argument`` is
the single declaration site for every knob and its default, and ``--help`` is
the full reference. There is no config file.

Usage:
    python scripts/run_step2.py --dataset esol --split-seed 0
    python scripts/run_step2.py --dataset esol --split-seed 0 --gpu-id 1 \
        --init-checkpoint experiments/step1/esol/seed_0/<ts>/model_0.pth
    python scripts/run_step2.py --dataset esol --split random --split-seed 0 \
        --init-checkpoint experiments/step1/esol/random/seed_0/<ts>/model_0.pth
    python scripts/run_step2.py --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_step1 import load_split, resolve_unimol_source  # noqa: E402
from src.data import DATASET_NAMES, get_dataset_info  # noqa: E402
from src.data.datasets import (  # noqa: E402
    DEFAULT_SPLIT,
    OUTPUT_DIR,
    SPLIT_TYPES,
    experiment_name,
)
from src.utils import Timer, print_banner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Declare every option, with its default, in one place."""
    p = argparse.ArgumentParser(
        prog='run_step2.py',
        description='Step 2: symbolic GP head + EGGROLL fine-tuning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset', required=True, choices=DATASET_NAMES,
                   default=argparse.SUPPRESS, help='dataset key from the registry')

    split = p.add_argument_group('data split')
    split.add_argument('--split', default=DEFAULT_SPLIT, choices=SPLIT_TYPES,
                       help='which split family to read; must match both '
                            'preprocess_data.py and --init-checkpoint')
    split.add_argument('--split-seed', type=int, default=0,
                       help='which split to train on')
    split.add_argument('--random-seed', type=int, default=42,
                       help='search seed; unrelated to --split-seed')
    split.add_argument('--n-folds', type=int, default=5,
                       help='CV folds inside train; this CV is the fitness for '
                            'both GP and ES. Always scaffold-grouped, even '
                            'under --split random, since its job is to stop the '
                            'search memorising scaffolds')

    head = p.add_argument_group('symbolic head')
    head.add_argument('--n-trees', type=int, default=16,
                      help='formula trees, one per embedding region')
    head.add_argument('--pop-size', type=int, default=200,
                      help='GP individuals per island')
    head.add_argument('--max-depth', type=int, default=5,
                      help='maximum tree depth')
    head.add_argument('--parsimony', type=float, default=5e-4,
                      help='CV-RMSE penalty per tree node')
    head.add_argument('--region-mode', default='contiguous',
                      choices=('contiguous', 'random'),
                      help='disjoint blocks, or overlapping random subspaces')
    head.add_argument('--no-probe', dest='use_probe', action='store_false',
                      default=True,
                      help='drop the linear-probe column; on by default so the '
                           'head contains the linear baseline as a special case '
                           '(use_probe default: %(default)s)')
    head.add_argument('--probe-penalty-scale', type=float, default=1.0,
                      help='extra ridge shrinkage on the probe column; >1 makes '
                           'the linear crutch costlier so trees must earn their keep')
    head.add_argument('--gp-replicas', type=int, default=1,
                      help='perturbed-backbone replicas the GP fitness averages '
                           'over; >1 selects trees robust across the ES '
                           'neighbourhood, at proportional CV cost')

    es = p.add_argument_group('evolution strategies (EGGROLL)')
    es.add_argument('--es-layers', type=int, default=4,
                    help='number of trailing transformer blocks ES perturbs')
    es.add_argument('--rank', type=int, default=4,
                    help='EGGROLL perturbation rank; keep pop-size*rank > 512 so '
                         'the aggregate update stays full rank, as in the paper')
    es.add_argument('--es-pop', type=int, default=256,
                    help='ES population per step (must be even for antithetic)')
    es.add_argument('--es-chunk', type=int, default=16,
                    help='members per forward pass; the main VRAM knob')
    es.add_argument('--mol-batch', type=int, default=256,
                    help='molecules scored per ES step')
    es.add_argument('--mol-tile', type=int, default=64,
                    help='molecules per length-sorted forward tile')
    es.add_argument('--sigma', type=float, default=3e-3,
                    help='perturbation scale, relative to each matrix norm')
    es.add_argument('--es-lr', type=float, default=1e-3,
                    help='Adam step, as a fraction of typical weight magnitude')
    es.add_argument('--shaping', default='zscore',
                    choices=('zscore', 'rank', 'zscore_rank'),
                    help='fitness shaping; zscore normalises error per molecule')
    es.add_argument('--no-antithetic', dest='antithetic', action='store_false',
                    default=True,
                    help='disable mirrored sampling '
                         '(antithetic default: %(default)s)')
    es.add_argument('--region-mask', action='store_true', default=False,
                    help='confine each perturbation column to one head region')
    es.add_argument('--es-dtype', default='fp32', choices=('fp32', 'bf16'),
                    help='suffix compute dtype; bf16 halves VRAM but only run it '
                         'after selftest passes in fp32')

    sched = p.add_argument_group('schedule')
    sched.add_argument('--warm-gens', type=int, default=300,
                       help='GP generations before the first ES phase')
    sched.add_argument('--phases', type=int, default=20,
                       help='ES/GP alternations')
    sched.add_argument('--es-steps', type=int, default=100,
                       help='ES steps per phase')
    sched.add_argument('--gp-gens', type=int, default=40,
                       help='GP generations per phase')
    sched.add_argument('--patience', type=int, default=5,
                       help='phases without validation improvement before stopping')

    init = p.add_argument_group('initialisation')
    init.add_argument('--init-checkpoint', default=None,
                      help='step-1 model_0.pth to start the ES mean from; '
                           'without it ES starts from the pretrained weights, '
                           'which is a much harder problem')
    init.add_argument('--remove-hs', action='store_true', default=False,
                      help='strip hydrogens and load the no-H checkpoint; must '
                           'match how --init-checkpoint was trained')

    hw = p.add_argument_group('hardware')
    hw.add_argument('--gpu-id', type=int, default=0, help='CUDA device index')
    hw.add_argument('--no-gpu', dest='use_gpu', action='store_false', default=True,
                    help='force CPU (use_gpu default: %(default)s)')

    out = p.add_argument_group('output')
    out.add_argument('--no-save', action='store_true',
                     help='run without writing results.json')
    return p


def set_clean_log_format() -> None:
    """Strip timestamps/levels from unimol_tools log lines."""
    fmt = logging.Formatter('%(message)s')
    for name in ('Uni-Mol Tools', 'unimol', ''):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(fmt)


def print_header(args, info, out_dir, unimol_dir, device, vram_note) -> None:
    print_banner('UniMol-GP -- Step 2: Symbolic Head + EGGROLL')
    rows = [
        ('Time', f'{datetime.now():%Y-%m-%d %H:%M:%S}'),
        ('Dataset', f"{args.dataset} ({info['task_type']}, {info['metric']})"),
        ('unimol_tools', unimol_dir),
        ('Device', device),
        ('Split / search seed', f'{args.split} {args.split_seed} / '
                                f'{args.random_seed}'),
        ('Head', f'{args.n_trees} trees x {512 // args.n_trees} dims, '
                 f'depth<={args.max_depth}, probe={args.use_probe}'),
        ('GP', f'pop={args.pop_size}/island, parsimony={args.parsimony}, '
               f'replicas={args.gp_replicas}'),
        ('ES', f'last {args.es_layers} layers, N={args.es_pop}, r={args.rank}, '
               f'sigma={args.sigma}, lr={args.es_lr}'),
        ('ES shaping', f'{args.shaping}, antithetic={args.antithetic}, '
                       f'region_mask={args.region_mask}'),
        ('Schedule', f'warm={args.warm_gens}, {args.phases} x '
                     f'({args.es_steps} ES + {args.gp_gens} GP)'),
        ('Init from', args.init_checkpoint or '(pretrained weights)'),
        ('VRAM estimate', vram_note),
        ('Save to', out_dir or '(--no-save)'),
    ]
    for label, value in rows:
        print(f'{label:<20}: {value}')


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)
    unimol_dir = resolve_unimol_source()

    import torch

    from src.es.data import MoleculeData
    from src.es.eggroll import ESConfig
    from src.es.forward_unimol import ESSpec, SplitUniMol
    from src.head.gp import GPConfig
    from src.train.stage2 import Stage2Config, Stage2Trainer

    if args.es_pop % 2 and args.antithetic:
        raise SystemExit('--es-pop must be even for antithetic sampling')
    if args.es_pop * args.rank <= 512:
        print(f'warning: es_pop*rank = {args.es_pop * args.rank} <= 512, so the '
              f'aggregate ES update is rank-deficient. Raise --rank or --es-pop.')

    device = torch.device(
        f'cuda:{args.gpu_id}' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_info = get_dataset_info(args.dataset)
    metric = dataset_info['metric']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = None if args.no_save else os.path.join(
        OUTPUT_DIR, experiment_name('step2', args.dataset, args.split_seed,
                                    args.split, timestamp))

    set_clean_log_format()
    from unimol_tools.models.nnmodel import OUTPUT_DIM
    from unimol_tools.models.unimol import UniMolModel

    # Step 2 never uses classification_head -- it reads the CLS representation
    # straight out of the encoder -- but the head still has to be the shape the
    # Step 1 checkpoint was saved with, or load_state_dict rejects the whole
    # file. strict=False does not help: it forgives missing and unexpected keys,
    # never a shape mismatch. Classification checkpoints carry a 2-way head.
    model = UniMolModel(output_dim=OUTPUT_DIM[dataset_info['task_type']],
                        data_type='molecule',
                        remove_hs=args.remove_hs).to(device).eval()
    if args.init_checkpoint:
        model.load_pretrained_weights(args.init_checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)

    spec = ESSpec(n_layers=args.es_layers, rank=args.rank,
                  region_mask=args.region_mask, n_regions=args.n_trees)
    dtype = torch.float32 if args.es_dtype == 'fp32' else torch.bfloat16
    split = SplitUniMol(model, spec, dtype=dtype)

    print_header(args, dataset_info, out_dir, unimol_dir, device,
                 '(after featurisation)')

    train_df, valid_df, test_df = load_split(args.dataset, args.split_seed,
                                             args.split)
    print(f'\nData -- Train: {len(train_df)}, Valid: {len(valid_df)}, '
          f'Test: {len(test_df)}')

    smi, tgt = 'smiles', 'target'
    splits = {}
    for name, df in (('train', train_df), ('valid', valid_df), ('test', test_df)):
        splits[name] = MoleculeData(
            df[smi].tolist(), df[tgt].to_numpy(), model,
            remove_hs=args.remove_hs, seed=args.random_seed,
        )
    # Valid and test must be standardised with the *training* statistics, or the
    # reported RMSE is measured on a different scale than the model was fit on.
    for name in ('valid', 'test'):
        data = splits[name]
        data.y_mean, data.y_std = splits['train'].y_mean, splits['train'].y_std
        data.scaled = (data.raw - data.y_mean) / data.y_std

    # Attention memory goes as S^2, and with all_h the longest molecule sets S
    # for its tile, so this can only be checked once the data is featurised.
    # Checked against *free* VRAM rather than card size: the card may be shared,
    # and an OOM four hours into an overnight sweep costs the whole night.
    seq = int(splits['train'].n_atoms.max())
    gb = split.attn_bytes(args.es_chunk, args.mol_tile, seq) * 3 / 1024 ** 3
    free = (torch.cuda.mem_get_info(device)[0] / 1024 ** 3
            if device.type == 'cuda' else float('inf'))
    print(f'Longest molecule: {seq} atoms -> ~{gb:.1f} GB attention peak '
          f'(es_chunk={args.es_chunk}, mol_tile={args.mol_tile}), '
          f'{free:.1f} GB free; length-sorted tiles make the typical tile '
          f'smaller.')
    if args.phases > 0 and gb > 0.8 * free:
        budget = 0.8 * free / max(gb, 1e-9)
        raise SystemExit(
            f'\nthat needs ~{gb:.1f} GB but only {free:.1f} GB is free.\n'
            f'Rerun with --es-chunk {max(1, int(args.es_chunk * budget))} '
            f'(or halve --mol-tile instead), or free the GPU first.\n')

    gp_cfg = GPConfig(
        n_trees=args.n_trees, pop_size=args.pop_size, max_depth=args.max_depth,
        parsimony=args.parsimony, use_probe=args.use_probe,
        probe_penalty_scale=args.probe_penalty_scale,
        region_mode=args.region_mode, replicas=args.gp_replicas,
    )
    es_cfg = ESConfig(
        pop_size=args.es_pop, pop_chunk=args.es_chunk, mol_batch=args.mol_batch,
        mol_tile=args.mol_tile, sigma=args.sigma, lr=args.es_lr,
        shaping=args.shaping, antithetic=args.antithetic, n_folds=args.n_folds,
        probe_penalty_scale=args.probe_penalty_scale, seed=args.random_seed,
    )
    stage_cfg = Stage2Config(
        warm_gens=args.warm_gens, n_phases=args.phases, es_steps=args.es_steps,
        gp_gens=args.gp_gens, patience=args.patience, seed=args.random_seed,
    )

    trainer = Stage2Trainer(
        split, splits['train'], splits['valid'], splits['test'],
        stage_cfg, gp_cfg, es_cfg, device,
        out_dir or os.path.join(OUTPUT_DIR, 'step2', '_scratch'),
        metric=metric,
    )
    with Timer(f'Step 2 {args.dataset} '
               f'({args.split} split_seed={args.split_seed})'):
        result = trainer.run()

    print('Learned formulas (best head):')
    for j, formula in enumerate(result['formulas']):
        print(f'  T{j:<2} [{result["tree_sizes"][j]:>2} nodes] {formula[:110]}')

    if out_dir:
        print(f'\nResults saved -- {os.path.join(out_dir, "results.json")}')
    print(f'End: {datetime.now():%Y-%m-%d %H:%M:%S}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

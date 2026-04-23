#!/usr/bin/env python
"""
Step 3 pre-compute: run UniMol (frozen with step 1 weights) on K conformers,
cache CLS embeddings per (dataset, split_seed, K).

Priority: CLI > config.yaml step3:* > DEFAULTS.

Cache path: {cache_dir}/{dataset}/seed_{s}/K_{K}.pt

Usage:
    # Standalone
    python scripts/precompute_embeddings.py --dataset esol
    python scripts/precompute_embeddings.py --dataset esol --split-seed 0 --K 10
    python scripts/precompute_embeddings.py --dataset esol --step1-weights path/to/model_0.pth --force

    # Auto-invoked from run_step3.py via run_precompute()
"""

import argparse
import logging
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import torch
import yaml

from src.data import DATASET_NAMES, get_dataset_info
from src.data.datasets import PROCESSED_DIR
from src.data.embeddings_cache import (
    build_and_save_cache,
    cache_exists,
    find_step1_weights,
    get_cache_path,
)

# ── Defaults ─────────────────────────────────────────────────────────────

DEFAULTS = {
    "split_seed":  0,
    "K":           10,
    "batch_size":  64,       # UniMol forward batch
    "gpu_id":      0,
    "use_gpu":     True,
    "conf_seed":   42,       # RDKit conformer generation seed
    "remove_hs":   True,
    "cache_dir":   "data/embeddings_cache",
}


# ── Config / params resolution ───────────────────────────────────────────

def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _pick(cli_val, cfg_val, default):
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default


def resolve_params(args, cfg):
    """CLI > config.step3.* / config.* > DEFAULTS."""
    s3cfg = cfg.get("step3", {}) or {}
    params = {}
    # n_confomer in step3 config block maps to K here
    params["K"]          = _pick(args.K, s3cfg.get("n_confomer"), DEFAULTS["K"])
    params["split_seed"] = _pick(args.split_seed, cfg.get("split_seed"), DEFAULTS["split_seed"])
    params["batch_size"] = _pick(args.batch_size, s3cfg.get("precompute_batch_size"), DEFAULTS["batch_size"])
    params["gpu_id"]     = _pick(args.gpu_id, cfg.get("gpu_id"), DEFAULTS["gpu_id"])
    params["use_gpu"]    = DEFAULTS["use_gpu"] if not args.no_gpu else False
    params["conf_seed"]  = _pick(args.conf_seed, s3cfg.get("conf_seed"), DEFAULTS["conf_seed"])
    params["remove_hs"]  = DEFAULTS["remove_hs"] if not args.no_remove_hs else False
    params["cache_dir"]  = _pick(args.cache_dir, s3cfg.get("embeddings_cache_dir"), DEFAULTS["cache_dir"])
    params["step1_weights"] = args.step1_weights  # explicit path or None (auto-latest)
    params["force"]      = args.force
    return params


# ── Data loading ─────────────────────────────────────────────────────────

def load_split(dataset_name, split_seed):
    seed_dir = os.path.join(PROCESSED_DIR, dataset_name, f"seed_{split_seed}")
    paths = {s: os.path.join(seed_dir, f"{dataset_name}_{s}.csv")
             for s in ("train", "valid", "test")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Data not found at {seed_dir}/\n"
            f"Run: python scripts/preprocess_data.py "
            f"--dataset {dataset_name} --split-seed {split_seed}"
        )
    return tuple(pd.read_csv(paths[s]) for s in ("train", "valid", "test"))


# ── Logging format ───────────────────────────────────────────────────────

def setup_logging(level=logging.INFO):
    """Minimal clean log format; silence UniMol's noisy loggers."""
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    # Silence UniMol tool internal logging
    for name in ("Uni-Mol Tools", "unimol_tools"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Core precompute (reusable from run_step3) ────────────────────────────

def run_precompute(params: dict, dataset_info: dict, verbose: bool = True) -> str:
    """
    Run full precompute pipeline. Reusable from run_step3.py's auto-invoke.

    Returns:
        cache_path (str) — written to or pre-existing.
    """
    dataset = dataset_info["name"]
    split_seed = params["split_seed"]
    K = params["K"]

    cache_path = get_cache_path(params["cache_dir"], dataset, split_seed, K)

    if cache_exists(params["cache_dir"], dataset, split_seed, K) and not params["force"]:
        if verbose:
            print(f"Cache already exists → {cache_path}")
            print(f"(Use --force to recompute.)")
        return cache_path

    # Resolve step 1 weights (explicit or auto-latest)
    weight_path, ts = find_step1_weights(
        dataset=dataset,
        split_seed=split_seed,
        explicit_path=params["step1_weights"],
    )
    if verbose:
        print(f"Step 1 weights: {weight_path}")
        print(f"Step 1 timestamp: {ts}")

    # Load data
    train_df, valid_df, test_df = load_split(dataset, split_seed)
    if verbose:
        print(f"Data -- train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Processed CSVs use standardized column names ('smiles', 'target') — this is the
    # same convention run_step2.py follows. The dataset_info's smiles/target_column fields
    # refer to RAW CSV columns (used by preprocess_data.py), NOT processed CSVs.
    SMILES_COL = "smiles"
    TARGET_COL = "target"
    for split_name, df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        for col in (SMILES_COL, TARGET_COL):
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in processed {split_name} CSV. "
                    f"Available columns: {list(df.columns)}. "
                    f"Re-run preprocess_data.py?"
                )

    # Device
    if params["use_gpu"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{params['gpu_id']}")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"Device: {device}")
        if dataset_info.get("task_type") != "regression":
            print(f"⚠ Note: dataset task_type={dataset_info.get('task_type')}. "
                  f"Step 3 scope is regression-only; precompute will still run.")

    # Build cache
    cache_path = build_and_save_cache(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        dataset=dataset,
        split_seed=split_seed,
        K=K,
        smiles_column=SMILES_COL,
        target_column=TARGET_COL,
        cache_dir=params["cache_dir"],
        device=device,
        step1_weight_path=weight_path,
        step1_timestamp=ts,
        batch_size=params["batch_size"],
        conf_seed=params["conf_seed"],
        remove_hs=params["remove_hs"],
    )

    return cache_path


# ── CLI entry ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Pre-compute UniMol embeddings cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",       type=str, required=True, choices=DATASET_NAMES)
    parser.add_argument("--split-seed",    type=int, default=None)
    parser.add_argument("--K",             type=int, default=None,
                        help="Target number of conformers per molecule")
    parser.add_argument("--batch-size",    type=int, default=None,
                        help="Batch size for UniMol forward (no_grad)")
    parser.add_argument("--gpu-id",        type=int, default=None)
    parser.add_argument("--no-gpu",        action="store_true")
    parser.add_argument("--conf-seed",     type=int, default=None,
                        help="Seed for RDKit conformer generation (determinism)")
    parser.add_argument("--no-remove-hs",  action="store_true",
                        help="Keep hydrogens (default: remove). Must match step 1.")
    parser.add_argument("--cache-dir",     type=str, default=None)
    parser.add_argument("--step1-weights", type=str, default=None,
                        help="Explicit path to step 1 model_0.pth. "
                             "If unset → auto-pick latest timestamp under "
                             "experiments/step1/{dataset}/seed_{s}/*/model_0.pth")
    parser.add_argument("--force",         action="store_true",
                        help="Recompute even if cache exists")
    parser.add_argument("--config",        type=str, default="config.yaml")

    args = parser.parse_args()
    os.chdir(project_root)
    cfg = load_config(args.config)
    params = resolve_params(args, cfg)
    dataset_info = get_dataset_info(args.dataset)

    setup_logging()

    # ── Header ───────────────────────────────────────
    print("=" * 70)
    print("UniMol-GP -- Step 3: Pre-compute Embeddings Cache")
    print("=" * 70)
    print(f"Time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Dataset     : {args.dataset} ({dataset_info['task_type']})")
    print(f"split_seed  : {params['split_seed']}")
    print(f"K           : {params['K']}")
    print(f"batch_size  : {params['batch_size']}")
    print(f"conf_seed   : {params['conf_seed']}")
    print(f"remove_hs   : {params['remove_hs']}")
    print(f"cache_dir   : {params['cache_dir']}")
    print(f"force       : {params['force']}")
    print(f"step1_wts   : {params['step1_weights'] or '(auto-latest)'}")
    print("=" * 70)

    try:
        cache_path = run_precompute(params, dataset_info, verbose=True)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        raise

    print(f"\n✓ Precompute done → {cache_path}")
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
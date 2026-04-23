#!/usr/bin/env python
"""
Step 3: UniMol (frozen) + Multi-tree GP + double ridge readout.

Priority: CLI > config.yaml step3:* > DEFAULTS.

Flow:
    1. Check embeddings cache; if missing → auto-invoke precompute
    2. Instantiate Step3Trainer
    3. Evolution loop with best-of-run tracking on valid
    4. Save results.json + best_individual.pt + formulas.txt

Usage:
    # Basic run (auto-precompute if needed)
    python scripts/run_step3.py --dataset esol

    # Custom hyperparameters
    python scripts/run_step3.py --dataset esol --K 5 --num-trees-per-conformer 15 --pop-size 800

    # Force recompute cache
    python scripts/run_step3.py --dataset esol --force-precompute

    # Point to specific step 1 weights
    python scripts/run_step3.py --dataset esol \\
        --step1-weights experiments/step1/esol/seed_0/20250115_120000/model_0.pth
"""

import argparse
import logging
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Scripts folder for run_precompute import
sys.path.insert(0, os.path.join(project_root, "scripts"))

import yaml  # noqa: E402

from src.data import DATASET_NAMES, get_dataset_info  # noqa: E402
from src.data.datasets import OUTPUT_DIR  # noqa: E402
from src.data.embeddings_cache import cache_exists, get_cache_path  # noqa: E402
from src.trainers.step3_gp import Step3Trainer  # noqa: E402

# Import precompute pipeline (reuse run_precompute function)
from precompute_embeddings import run_precompute as run_precompute_embeddings  # noqa: E402


# ── ALL defaults ─────────────────────────────────────────────────────────

DEFAULTS = {
    # Shared
    "split_seed":             0,
    "random_seed":            42,
    "gpu_id":                 0,
    "use_gpu":                True,

    # Cache / precompute
    "cache_dir":              "data/embeddings_cache",
    "step1_weights":          None,
    "precompute_batch_size":  64,
    "conf_seed":              42,
    "remove_hs":              True,

    # GP architecture
    "K":                      10,
    "num_trees_per_conformer": 10,
    "D":                      64,           # ⚠ MUST be ≤ evogp CUDA stack limit (≈64)
    "pop_size":               500,
    "max_tree_len":           640,
    "max_layer_cnt":          5,
    "mutation_max_layer_cnt": 3,
    "using_funcs":            ["+", "-", "*", "/"],
    "const_samples":          [-1.0, -0.5, 0.0, 0.5, 1.0],
    "mutation_rate":          0.2,
    "survival_rate":          0.3,
    "elite_rate":             0.01,
    "parsimony_alpha":        0.0,

    # Ridge
    "lambda_inner":           1.0,
    "lambda_outer":           0.1,

    # Training loop
    "num_generations":        100,
    "patience":               15,
    "no_save":                False,
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
    s3 = cfg.get("step3", {}) or {}
    p = {}

    # ── Shared (top-level cfg) ──
    p["split_seed"]   = _pick(args.split_seed, cfg.get("split_seed"), DEFAULTS["split_seed"])
    p["random_seed"]  = _pick(args.random_seed, cfg.get("random_seed"), DEFAULTS["random_seed"])
    p["gpu_id"]       = _pick(args.gpu_id, cfg.get("gpu_id"), DEFAULTS["gpu_id"])
    p["use_gpu"]      = DEFAULTS["use_gpu"] if not args.no_gpu else False

    # ── Cache / precompute ──
    p["cache_dir"]             = _pick(args.cache_dir, s3.get("embeddings_cache_dir"), DEFAULTS["cache_dir"])
    p["step1_weights"]         = args.step1_weights  # CLI only (explicit override)
    p["precompute_batch_size"] = _pick(None, s3.get("precompute_batch_size"), DEFAULTS["precompute_batch_size"])
    p["conf_seed"]             = _pick(None, s3.get("conf_seed"), DEFAULTS["conf_seed"])
    p["remove_hs"]             = DEFAULTS["remove_hs"]  # must match step 1

    # ── GP: K aliased from step3.n_confomer in config ──
    p["K"]                     = _pick(args.K, s3.get("n_confomer"), DEFAULTS["K"])
    # Accept both new name and legacy 'q' in config for backward compat
    ntp_cfg = s3.get("num_trees_per_conformer", s3.get("q"))
    p["num_trees_per_conformer"] = _pick(args.num_trees_per_conformer, ntp_cfg,
                                          DEFAULTS["num_trees_per_conformer"])
    p["D"]                     = _pick(args.gp_input_dim, s3.get("gp_input_dim"), DEFAULTS["D"])
    p["pop_size"]              = _pick(args.pop_size, s3.get("pop_size"), DEFAULTS["pop_size"])
    p["max_tree_len"]          = _pick(args.max_tree_len, s3.get("max_tree_len"), DEFAULTS["max_tree_len"])
    p["max_layer_cnt"]         = _pick(args.max_layer_cnt, s3.get("max_layer_cnt"), DEFAULTS["max_layer_cnt"])
    p["mutation_max_layer_cnt"]= _pick(None, s3.get("mutation_max_layer_cnt"), DEFAULTS["mutation_max_layer_cnt"])
    p["using_funcs"]           = _pick(None, s3.get("using_funcs"), DEFAULTS["using_funcs"])
    p["const_samples"]         = _pick(None, s3.get("const_samples"), DEFAULTS["const_samples"])
    p["mutation_rate"]         = _pick(None, s3.get("mutation_rate"), DEFAULTS["mutation_rate"])
    p["survival_rate"]         = _pick(None, s3.get("survival_rate"), DEFAULTS["survival_rate"])
    p["elite_rate"]            = _pick(None, s3.get("elite_rate"), DEFAULTS["elite_rate"])
    p["parsimony_alpha"]       = _pick(args.parsimony_alpha, s3.get("parsimony_alpha"), DEFAULTS["parsimony_alpha"])

    # ── Ridge ──
    p["lambda_inner"]          = _pick(args.lambda_inner, s3.get("lambda_inner"), DEFAULTS["lambda_inner"])
    p["lambda_outer"]          = _pick(args.lambda_outer, s3.get("lambda_outer"), DEFAULTS["lambda_outer"])

    # ── Training ──
    p["num_generations"]       = _pick(args.num_generations, s3.get("num_generations"), DEFAULTS["num_generations"])
    p["patience"]              = _pick(args.patience, s3.get("patience"), DEFAULTS["patience"])
    p["no_save"]               = args.no_save

    # ── Precompute control (CLI only) ──
    p["force_precompute"]      = args.force_precompute

    return p


# ── Logging ──────────────────────────────────────────────────────────────

def setup_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    for name in ("Uni-Mol Tools", "unimol_tools"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: UniMol + Multi-tree GP + Ridge training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",      type=str, required=True, choices=DATASET_NAMES)

    # Shared
    parser.add_argument("--split-seed",   type=int, default=None)
    parser.add_argument("--random-seed",  type=int, default=None)
    parser.add_argument("--gpu-id",       type=int, default=None)
    parser.add_argument("--no-gpu",       action="store_true")

    # GP architecture
    parser.add_argument("--K",            type=int, default=None,
                        help="Number of conformers per molecule")
    parser.add_argument("--num-trees-per-conformer", type=int, default=None,
                        dest="num_trees_per_conformer",
                        help="Number of GP trees per conformer (Forest output_len)")
    parser.add_argument("--gp-input-dim", type=int, default=None,
                        help="PCA-reduced dim fed into GP (default 64). "
                             "MUST be ≲64 due to evogp CUDA stack limit.")
    parser.add_argument("--pop-size",     type=int, default=None)
    parser.add_argument("--max-tree-len", type=int, default=None,
                        help="Max nodes per individual (shared across q subtrees). "
                             "Rule of thumb: q × (2^(max_layer_cnt+1) - 1)")
    parser.add_argument("--max-layer-cnt",type=int, default=None,
                        help="Max depth per subtree")
    parser.add_argument("--parsimony-alpha", type=float, default=None,
                        help="Tree size penalty coefficient (0 = disabled)")

    # Ridge
    parser.add_argument("--lambda-inner", type=float, default=None,
                        help="L2 regularization for per-conformer inner ridge")
    parser.add_argument("--lambda-outer", type=float, default=None,
                        help="L2 regularization for outer ridge across K conformers")

    # Training loop
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--patience",     type=int, default=None,
                        help="Generations without valid improvement before early stop")

    # Cache / precompute
    parser.add_argument("--cache-dir",    type=str, default=None)
    parser.add_argument("--step1-weights",type=str, default=None,
                        help="Explicit path to step 1 model_0.pth. "
                             "If unset → auto-pick latest timestamp.")
    parser.add_argument("--force-precompute", action="store_true",
                        help="Force recompute embeddings cache even if exists")

    # Save
    parser.add_argument("--no-save",      action="store_true",
                        help="Skip writing artifacts (results.json, best_individual.pt, formulas.txt)")
    parser.add_argument("--config",       type=str, default="config.yaml")

    args = parser.parse_args()
    os.chdir(project_root)
    cfg = load_config(args.config)
    params = resolve_params(args, cfg)
    dataset_info = get_dataset_info(args.dataset)

    setup_logging()

    split_seed = params["split_seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Header ───────────────────────────────────────────────────
    print("=" * 70)
    print("UniMol-GP -- Step 3: Multi-tree GP + Ridge Training")
    print("=" * 70)
    print(f"Time         : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Dataset      : {args.dataset} ({dataset_info['task_type']})")
    print(f"split_seed   : {split_seed}")
    print(f"random_seed  : {params['random_seed']}")
    print(f"gpu_id       : {params['gpu_id']}")
    print(f"--- GP ---")
    print(f"K                       : {params['K']}")
    print(f"num_trees_per_conformer : {params['num_trees_per_conformer']}")
    print(f"D (PCA)                 : {params['D']} (from 512-dim CLS)")
    print(f"pop_size                : {params['pop_size']}")
    print(f"max_tree_len            : {params['max_tree_len']}")
    print(f"max_layer_cnt           : {params['max_layer_cnt']} (mutation uses {params['mutation_max_layer_cnt']})")
    print(f"parsimony_α             : {params['parsimony_alpha']}")
    print(f"ops                     : {params['using_funcs']}")
    print(f"--- Ridge ---")
    print(f"λ_inner                 : {params['lambda_inner']}")
    print(f"λ_outer                 : {params['lambda_outer']}")
    print(f"--- Training ---")
    print(f"num_gens                : {params['num_generations']}")
    print(f"patience                : {params['patience']}")
    if not params["no_save"]:
        print(f"Save to                 : {OUTPUT_DIR}/step3/{args.dataset}/seed_{split_seed}/{timestamp}/")
    print("=" * 70)

    # ── Scope guard: regression only ─────────────────────────────
    if dataset_info["task_type"] != "regression":
        print(f"\n✗ Error: Step 3 scope is regression only, got "
              f"task_type={dataset_info['task_type']} for dataset {args.dataset}")
        sys.exit(1)

    # ── Cache check + auto-invoke precompute ─────────────────────
    K = params["K"]
    cache_dir = params["cache_dir"]
    have_cache = cache_exists(cache_dir, args.dataset, split_seed, K)

    if have_cache and not params["force_precompute"]:
        cache_path = get_cache_path(cache_dir, args.dataset, split_seed, K)
        print(f"\n✓ Using existing cache → {cache_path}")
    else:
        if params["force_precompute"]:
            print("\n[--force-precompute set: recomputing embeddings cache]")
        else:
            print(f"\n[Cache not found → auto-invoking precompute]")

        precompute_params = {
            "K":             K,
            "split_seed":    split_seed,
            "batch_size":    params["precompute_batch_size"],
            "gpu_id":        params["gpu_id"],
            "use_gpu":       params["use_gpu"],
            "conf_seed":     params["conf_seed"],
            "remove_hs":     params["remove_hs"],
            "cache_dir":     cache_dir,
            "step1_weights": params["step1_weights"],
            "force":         params["force_precompute"],
        }
        try:
            cache_path = run_precompute_embeddings(precompute_params, dataset_info, verbose=True)
        except FileNotFoundError as e:
            print(f"\n✗ Precompute failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Unexpected error during precompute: {type(e).__name__}: {e}")
            raise

    # ── Run Step3Trainer ─────────────────────────────────────────
    experiment_name = f"step3/{args.dataset}/seed_{split_seed}/{timestamp}"

    trainer = Step3Trainer(
        params=params,
        dataset_info=dataset_info,
        experiment_name=experiment_name,
    )

    try:
        results = trainer.run(cache_path)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Training error: {type(e).__name__}: {e}")
        raise

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best gen       : {results['best_gen']}")
    print(f"Train RMSE     : {results['train_rmse']:.4f}  (MSE={results['train_mse']:.4f})")
    print(f"Valid RMSE     : {results['valid_rmse']:.4f}  (MSE={results['valid_mse']:.4f})")
    print(f"Test  RMSE     : {results['test_rmse']:.4f}  (MSE={results['test_mse']:.4f})  ← MAIN")
    print(f"Total time     : {results['total_time_s']:.1f}s "
          f"({results['num_generations_run']} generations)")
    if not params["no_save"]:
        print(f"Saved to       : {OUTPUT_DIR}/{experiment_name}/")
        print(f"  - results.json")
        print(f"  - best_individual.pt")
        print(f"  - formulas.txt")
    print("=" * 70)
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
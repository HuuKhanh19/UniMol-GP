#!/usr/bin/env python
"""
Step 3: 2-phase pipeline
  Phase 1: Train UniMol with K conformers (multi-conf, random duplication)
  Phase 2: Multi-tree GP + double ridge readout on Phase 1 embeddings

Priority: CLI > config.yaml step3:* > DEFAULTS.

Flow:
    1. Check embeddings cache:
       - if exists and not --force-phase1: skip Phase 1, load cache
       - else: run Phase 1 (train UniMol + extract embeddings → cache)
    2. Run Phase 2 GP training with cache
    3. Save phase1_results.json (Phase 1) + results.json (Phase 2)
       + best_individual.pt + formulas.txt

Usage:
    python scripts/run_step3.py --dataset esol
    python scripts/run_step3.py --dataset esol --force-phase1
    python scripts/run_step3.py --dataset esol --K 5 --num-trees-per-conformer 15 --pop-size 800
"""

# ──────────────────────────────────────────────────────────────────────
# IMPORTANT: Set env vars BEFORE importing torch for hard determinism.
# These must be set before any CUDA initialization happens.
# ──────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic CUBLAS
os.environ.setdefault("PYTHONHASHSEED", "42")                # deterministic hashing

import argparse
import json
import logging
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import DATASET_NAMES, get_dataset_info  # noqa: E402
from src.data.datasets import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.data.embeddings_cache import (  # noqa: E402
    assemble_cache_dict,
    cache_exists,
    get_cache_path,
    save_cache,
)
from src.data.multi_conf_dataset import (  # noqa: E402
    IndexedMolKConfDataset,
    MolKConfDataset,
)
from src.trainers.step3_gp import Step3Trainer  # noqa: E402
from src.trainers.step3_phase1_train import Phase1Config, Phase1Trainer  # noqa: E402


# ── ALL defaults ─────────────────────────────────────────────────────────

DEFAULTS = {
    # Shared
    "split_seed":             0,
    "random_seed":            42,
    "gpu_id":                 0,
    "use_gpu":                True,

    # Cache
    "cache_dir":              "data/embeddings_cache",
    "remove_hs":              True,
    "conf_seed":              42,

    # Phase 1 (UniMol training)
    "phase1_epochs":          50,
    "phase1_batch_size":      16,
    "phase1_learning_rate":   1.0e-4,
    "phase1_patience":        10,
    "phase1_warmup_ratio":    0.03,
    "phase1_max_norm":        5.0,
    "phase1_use_amp":         True,
    "phase1_weight_decay":    0.0,
    "phase1_target_normalize": "auto",

    # GP architecture (Phase 2)
    "K":                      10,
    "num_trees_per_conformer": 10,
    "D":                      64,
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

    # Phase 2 GP loop
    "num_generations":        100,
    "patience":               15,

    # Save control
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
    ph1 = s3.get("phase1", {}) or {}
    p = {}

    # Shared
    p["split_seed"]   = _pick(args.split_seed, cfg.get("split_seed"), DEFAULTS["split_seed"])
    p["random_seed"]  = _pick(args.random_seed, cfg.get("random_seed"), DEFAULTS["random_seed"])
    p["gpu_id"]       = _pick(args.gpu_id, cfg.get("gpu_id"), DEFAULTS["gpu_id"])
    p["use_gpu"]      = DEFAULTS["use_gpu"] if not args.no_gpu else False

    # Cache
    p["cache_dir"] = _pick(args.cache_dir, s3.get("embeddings_cache_dir"), DEFAULTS["cache_dir"])
    p["remove_hs"] = DEFAULTS["remove_hs"]
    p["conf_seed"] = _pick(None, s3.get("conf_seed"), DEFAULTS["conf_seed"])

    # Phase 1
    p["phase1_epochs"]          = _pick(args.phase1_epochs, ph1.get("epochs"), DEFAULTS["phase1_epochs"])
    p["phase1_batch_size"]      = _pick(args.phase1_batch_size, ph1.get("batch_size"), DEFAULTS["phase1_batch_size"])
    p["phase1_learning_rate"]   = _pick(args.phase1_learning_rate, ph1.get("learning_rate"), DEFAULTS["phase1_learning_rate"])
    p["phase1_patience"]        = _pick(args.phase1_patience, ph1.get("patience"), DEFAULTS["phase1_patience"])
    p["phase1_warmup_ratio"]    = _pick(None, ph1.get("warmup_ratio"), DEFAULTS["phase1_warmup_ratio"])
    p["phase1_max_norm"]        = _pick(None, ph1.get("max_norm"), DEFAULTS["phase1_max_norm"])
    p["phase1_use_amp"]         = _pick(None, ph1.get("use_amp"), DEFAULTS["phase1_use_amp"])
    p["phase1_weight_decay"]    = _pick(None, ph1.get("weight_decay"), DEFAULTS["phase1_weight_decay"])
    p["phase1_target_normalize"] = _pick(None, ph1.get("target_normalize"), DEFAULTS["phase1_target_normalize"])

    # GP
    p["K"]                       = _pick(args.K, s3.get("n_confomer"), DEFAULTS["K"])
    ntp_cfg                      = s3.get("num_trees_per_conformer", s3.get("q"))
    p["num_trees_per_conformer"] = _pick(args.num_trees_per_conformer, ntp_cfg, DEFAULTS["num_trees_per_conformer"])
    p["D"]                       = _pick(args.gp_input_dim, s3.get("gp_input_dim"), DEFAULTS["D"])
    p["pop_size"]                = _pick(args.pop_size, s3.get("pop_size"), DEFAULTS["pop_size"])
    p["max_tree_len"]            = _pick(args.max_tree_len, s3.get("max_tree_len"), DEFAULTS["max_tree_len"])
    p["max_layer_cnt"]           = _pick(args.max_layer_cnt, s3.get("max_layer_cnt"), DEFAULTS["max_layer_cnt"])
    p["mutation_max_layer_cnt"]  = _pick(None, s3.get("mutation_max_layer_cnt"), DEFAULTS["mutation_max_layer_cnt"])
    p["using_funcs"]             = _pick(None, s3.get("using_funcs"), DEFAULTS["using_funcs"])
    p["const_samples"]           = _pick(None, s3.get("const_samples"), DEFAULTS["const_samples"])
    p["mutation_rate"]           = _pick(None, s3.get("mutation_rate"), DEFAULTS["mutation_rate"])
    p["survival_rate"]           = _pick(None, s3.get("survival_rate"), DEFAULTS["survival_rate"])
    p["elite_rate"]              = _pick(None, s3.get("elite_rate"), DEFAULTS["elite_rate"])
    p["parsimony_alpha"]         = _pick(args.parsimony_alpha, s3.get("parsimony_alpha"), DEFAULTS["parsimony_alpha"])

    # Ridge
    p["lambda_inner"] = _pick(args.lambda_inner, s3.get("lambda_inner"), DEFAULTS["lambda_inner"])
    p["lambda_outer"] = _pick(args.lambda_outer, s3.get("lambda_outer"), DEFAULTS["lambda_outer"])

    # GP loop
    p["num_generations"] = _pick(args.num_generations, s3.get("num_generations"), DEFAULTS["num_generations"])
    p["patience"]        = _pick(args.patience, s3.get("patience"), DEFAULTS["patience"])

    p["no_save"]         = args.no_save
    p["force_phase1"]    = args.force_phase1

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


def generate_conformers_for_split(
    smiles_list,
    K: int,
    seed: int = 42,
    remove_hs: bool = True,
    desc: str = "split",
):
    """Generate up to K conformers per molecule using ConformerGen."""
    from unimol_tools.data.conformer import ConformerGen
    from tqdm import tqdm

    cg = ConformerGen(
        data_type="molecule",
        remove_hs=remove_hs,
        n_confomer=K,
        seed=seed,
        multi_process=False,
    )

    conformers_per_mol = []
    for smi in tqdm(smiles_list, desc=f"Generating K={K} confs ({desc})"):
        try:
            feats = cg.single_process(smi)
        except Exception as e:
            print(f"  ConformerGen failed for SMILES: {e}")
            feats = []
        # Filter conformers with all-zero src_coord (RDKit fail)
        valid_feats = [f for f in feats if not (f["src_coord"] == 0.0).all()]
        conformers_per_mol.append(valid_feats)

    return conformers_per_mol


# ── Phase 1 orchestration ────────────────────────────────────────────────

def run_phase1(params, dataset_info, output_dir):
    """
    Run Phase 1: train UniMol with K confs, extract embeddings cache.
    Returns: (cache_path, phase1_results_dict)
    """
    dataset = dataset_info["name"]
    split_seed = params["split_seed"]
    K = params["K"]

    use_gpu = params["use_gpu"] and torch.cuda.is_available()
    device = torch.device(f"cuda:{params['gpu_id']}") if use_gpu else torch.device("cpu")
    print(f"\nDevice: {device}")

    # Load CSVs
    train_df, valid_df, test_df = load_split(dataset, split_seed)
    print(f"Data -- train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Determine task type and num_tasks
    task_type = dataset_info["task_type"]
    target_col = "target"
    train_targets = train_df[target_col].values
    if train_targets.ndim == 1:
        num_tasks = 1
    else:
        num_tasks = train_targets.shape[1]
    print(f"Task type: {task_type} | num_tasks: {num_tasks}")

    # Generate conformers
    print("\n=== Generating conformers ===")
    train_confs = generate_conformers_for_split(
        train_df["smiles"].tolist(), K=K, seed=params["conf_seed"],
        remove_hs=params["remove_hs"], desc="train"
    )
    valid_confs = generate_conformers_for_split(
        valid_df["smiles"].tolist(), K=K, seed=params["conf_seed"],
        remove_hs=params["remove_hs"], desc="valid"
    )
    test_confs = generate_conformers_for_split(
        test_df["smiles"].tolist(), K=K, seed=params["conf_seed"],
        remove_hs=params["remove_hs"], desc="test"
    )

    # Build datasets (filter molecules with 0 conformers)
    train_targets_arr = np.asarray(train_df[target_col].values, dtype=np.float32)
    valid_targets_arr = np.asarray(valid_df[target_col].values, dtype=np.float32)
    test_targets_arr  = np.asarray(test_df[target_col].values, dtype=np.float32)

    train_ds = MolKConfDataset(train_confs, train_targets_arr)
    valid_ds = IndexedMolKConfDataset(valid_confs, valid_targets_arr)
    test_ds  = IndexedMolKConfDataset(test_confs,  test_targets_arr)

    print(f"\nDatasets after filtering: "
          f"train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    print(f"Valid counts (train): "
          f"min={train_ds.valid_counts.min()}, "
          f"mean={train_ds.valid_counts.mean():.2f}, "
          f"max={train_ds.valid_counts.max()}")

    # Build Phase 1 config
    ph1_cfg = Phase1Config(
        K=K,
        epochs=params["phase1_epochs"],
        batch_size=params["phase1_batch_size"],
        learning_rate=params["phase1_learning_rate"],
        patience=params["phase1_patience"],
        warmup_ratio=params["phase1_warmup_ratio"],
        max_norm=params["phase1_max_norm"],
        use_amp=params["phase1_use_amp"],
        weight_decay=params["phase1_weight_decay"],
        target_normalize=params["phase1_target_normalize"],
        training_seed=params["random_seed"],
        fixed_seed_eval=0,
    )

    # Run Phase 1
    trainer = Phase1Trainer(
        cfg=ph1_cfg, task_type=task_type, num_tasks=num_tasks, device=device,
    )
    phase1_output_dir = os.path.join(output_dir, "phase1")
    os.makedirs(phase1_output_dir, exist_ok=True)
    phase1_results = trainer.run(train_ds, valid_ds, test_ds, output_dir=phase1_output_dir)

    # Save phase1_results.json
    if not params.get("no_save", False):
        history_path = os.path.join(phase1_output_dir, "phase1_results.json")
        with open(history_path, "w", encoding="utf-8") as f:
            serializable = {
                "best_epoch": phase1_results["best_epoch"],
                "best_valid_metric": phase1_results["best_valid_metric"],
                "history": phase1_results["history"],
                "model_path": phase1_results["model_path"],
                "metrics": phase1_results["metrics"],
                "total_time_s": phase1_results["total_time_s"],
                "target_mean": phase1_results["target_mean"],
                "target_std": phase1_results["target_std"],
            }
            json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)
        print(f"\nPhase 1 results saved → {history_path}")

    # Assemble cache
    cache_dict = assemble_cache_dict(
        cache_train=phase1_results["cache_train"],
        cache_valid=phase1_results["cache_valid"],
        cache_test=phase1_results["cache_test"],
        dataset=dataset,
        split_seed=split_seed,
        K=K,
        phase1_model_path=phase1_results["model_path"],
        phase1_timestamp=os.path.basename(output_dir),
        remove_hs=params["remove_hs"],
    )

    cache_path = get_cache_path(params["cache_dir"], dataset, split_seed, K)
    save_cache(cache_path, cache_dict)

    return cache_path, phase1_results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Phase 1 (UniMol multi-conf training) + Phase 2 (GP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",      type=str, required=True, choices=DATASET_NAMES)

    # Shared
    parser.add_argument("--split-seed",   type=int, default=None)
    parser.add_argument("--random-seed",  type=int, default=None)
    parser.add_argument("--gpu-id",       type=int, default=None)
    parser.add_argument("--no-gpu",       action="store_true")

    # GP architecture
    parser.add_argument("--K",            type=int, default=None)
    parser.add_argument("--num-trees-per-conformer", type=int, default=None,
                        dest="num_trees_per_conformer")
    parser.add_argument("--gp-input-dim", type=int, default=None)
    parser.add_argument("--pop-size",     type=int, default=None)
    parser.add_argument("--max-tree-len", type=int, default=None)
    parser.add_argument("--max-layer-cnt",type=int, default=None)
    parser.add_argument("--parsimony-alpha", type=float, default=None)

    # Ridge
    parser.add_argument("--lambda-inner", type=float, default=None)
    parser.add_argument("--lambda-outer", type=float, default=None)

    # GP loop
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--patience",     type=int, default=None)

    # Phase 1
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase1-batch-size", type=int, default=None)
    parser.add_argument("--phase1-learning-rate", type=float, default=None)
    parser.add_argument("--phase1-patience", type=int, default=None)

    # Cache / save
    parser.add_argument("--cache-dir",    type=str, default=None)
    parser.add_argument("--force-phase1", action="store_true")
    parser.add_argument("--no-save",      action="store_true")
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
    print("UniMol-GP -- Step 3: Phase 1 (UniMol Multi-Conf) + Phase 2 (GP)")
    print("=" * 70)
    print(f"Time         : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Dataset      : {args.dataset} ({dataset_info['task_type']})")
    print(f"split_seed   : {split_seed}")
    print(f"random_seed  : {params['random_seed']}")
    print(f"gpu_id       : {params['gpu_id']}")
    print(f"--- Phase 1 (UniMol training) ---")
    print(f"K                       : {params['K']}")
    print(f"epochs                  : {params['phase1_epochs']}")
    print(f"batch_size              : {params['phase1_batch_size']}")
    print(f"learning_rate           : {params['phase1_learning_rate']}")
    print(f"patience                : {params['phase1_patience']}")
    print(f"--- Phase 2 (GP) ---")
    print(f"num_trees_per_conformer : {params['num_trees_per_conformer']}")
    print(f"D (PCA)                 : {params['D']}")
    print(f"pop_size                : {params['pop_size']}")
    print(f"max_tree_len            : {params['max_tree_len']}")
    print(f"max_layer_cnt           : {params['max_layer_cnt']} (mutation uses {params['mutation_max_layer_cnt']})")
    print(f"ops                     : {params['using_funcs']}")
    print(f"--- Ridge ---")
    print(f"λ_inner                 : {params['lambda_inner']}")
    print(f"λ_outer                 : {params['lambda_outer']}")
    print(f"--- GP loop ---")
    print(f"num_gens                : {params['num_generations']}")
    print(f"patience                : {params['patience']}")
    if not params["no_save"]:
        print(f"Save to                 : {OUTPUT_DIR}/step3/{args.dataset}/seed_{split_seed}/{timestamp}/")
    print("=" * 70)

    # ── Scope guard: regression only (Phase 2 GP) ────────────────
    if dataset_info["task_type"] != "regression":
        print(f"\n✗ Error: Phase 2 GP scope is regression only, got "
              f"task_type={dataset_info['task_type']} for dataset {args.dataset}")
        print("   (Phase 1 supports all task types, but PR 3 hasn't extended Phase 2 yet.)")
        sys.exit(1)

    # ── Cache check + Phase 1 ────────────────────────────────────
    K = params["K"]
    cache_dir = params["cache_dir"]
    have_cache = cache_exists(cache_dir, args.dataset, split_seed, K)

    output_dir = os.path.join(OUTPUT_DIR, "step3", args.dataset,
                               f"seed_{split_seed}", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    if have_cache and not params["force_phase1"]:
        cache_path = get_cache_path(cache_dir, args.dataset, split_seed, K)
        print(f"\n✓ Using existing cache → {cache_path}")
        print(f"   (Use --force-phase1 to re-run Phase 1.)")
        phase1_results = None
    else:
        if params["force_phase1"]:
            print("\n[--force-phase1: re-running Phase 1]")
        else:
            print(f"\n[Cache not found → running Phase 1]")
        try:
            cache_path, phase1_results = run_phase1(params, dataset_info, output_dir)
        except Exception as e:
            print(f"\n✗ Phase 1 error: {type(e).__name__}: {e}")
            raise

    # ── Phase 2: GP training ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Multi-tree GP training")
    print("=" * 70)

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
        print(f"\n✗ Phase 2 error: {type(e).__name__}: {e}")
        raise

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    if phase1_results is not None:
        m = phase1_results["metrics"]
        print(f"--- Phase 1 (UniMol multi-conf) ---")
        print(f"Best epoch     : {phase1_results['best_epoch']}")
        for split in ("train", "valid", "test"):
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in m[split].items())
            print(f"{split.capitalize():5s} | {metric_str}")
    print(f"--- Phase 2 (GP) ---")
    print(f"Best gen       : {results['best_gen']}")
    print(f"Train RMSE     : {results['train_rmse']:.4f}  (MSE={results['train_mse']:.4f})")
    print(f"Valid RMSE     : {results['valid_rmse']:.4f}  (MSE={results['valid_mse']:.4f})")
    print(f"Test  RMSE     : {results['test_rmse']:.4f}  (MSE={results['test_mse']:.4f})  ← MAIN")
    print(f"Total time     : {results['total_time_s']:.1f}s "
          f"({results['num_generations_run']} generations)")
    if not params["no_save"]:
        print(f"Saved to       : {OUTPUT_DIR}/{experiment_name}/")
    print("=" * 70)
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
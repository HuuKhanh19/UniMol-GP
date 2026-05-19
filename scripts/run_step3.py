#!/usr/bin/env python
"""
Step 3: 3-phase pipeline
  Phase 1: Train UniMol with K conformers (multi-conf, random duplication)
  Phase 2: Multi-tree GP + double ridge readout on Phase 1 embeddings
  Phase 3: SGD fine-tune encoder with frozen GP+ridge readout (regression only)

Priority: CLI > config.yaml step3:* > DEFAULTS.

Flow:
    1. Resolve params (CLI / config / defaults)
    2. Build datasets (only if needed: cache miss OR Phase 3 enabled)
    3. Phase 1: run if cache miss or --force-phase1; else load cache
    4. Phase 2: GP training, save best_individual.pt
    5. Phase 3: encoder fine-tune (if enabled & regression)
    6. Final summary

Usage:
    python scripts/run_step3.py --dataset esol
    python scripts/run_step3.py --dataset esol --force-phase1
    python scripts/run_step3.py --dataset esol --phase3-enabled false
    python scripts/run_step3.py --dataset esol --phase3-epochs 20 --phase3-lambda-anchor 1.0
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


# ──────────────────────────────────────────────────────────────────────
# ALL defaults
# ──────────────────────────────────────────────────────────────────────

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

    # Phase 3 (encoder fine-tune)
    "phase3_enabled":         True,
    "phase3_epochs":          15,
    "phase3_batch_size":      4,
    "phase3_learning_rate":   1.0e-5,
    "phase3_patience":        5,
    "phase3_max_norm":        5.0,
    "phase3_use_amp":         False,
    "phase3_lambda_anchor":   0.5,
    "phase3_pipeline_mode":   "safe",   # "safe" | "vanilla"
    "phase3_random_dup_train": True,

    # Save control
    "no_save":                False,
}


# ──────────────────────────────────────────────────────────────────────
# Config / params resolution
# ──────────────────────────────────────────────────────────────────────

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


def _str_to_bool(s):
    """Parse string '--flag true/false' values from CLI."""
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("true", "1", "yes", "y")


def resolve_params(args, cfg):
    """CLI > config.step3.* / config.* > DEFAULTS."""
    s3 = cfg.get("step3", {}) or {}
    ph1 = s3.get("phase1", {}) or {}
    ph3 = s3.get("phase3", {}) or {}
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
    p["phase1_epochs"]           = _pick(args.phase1_epochs, ph1.get("epochs"), DEFAULTS["phase1_epochs"])
    p["phase1_batch_size"]       = _pick(args.phase1_batch_size, ph1.get("batch_size"), DEFAULTS["phase1_batch_size"])
    p["phase1_learning_rate"]    = _pick(args.phase1_learning_rate, ph1.get("learning_rate"), DEFAULTS["phase1_learning_rate"])
    p["phase1_patience"]         = _pick(args.phase1_patience, ph1.get("patience"), DEFAULTS["phase1_patience"])
    p["phase1_warmup_ratio"]     = _pick(None, ph1.get("warmup_ratio"), DEFAULTS["phase1_warmup_ratio"])
    p["phase1_max_norm"]         = _pick(None, ph1.get("max_norm"), DEFAULTS["phase1_max_norm"])
    p["phase1_use_amp"]          = _pick(None, ph1.get("use_amp"), DEFAULTS["phase1_use_amp"])
    p["phase1_weight_decay"]     = _pick(None, ph1.get("weight_decay"), DEFAULTS["phase1_weight_decay"])
    p["phase1_target_normalize"] = _pick(None, ph1.get("target_normalize"), DEFAULTS["phase1_target_normalize"])

    # GP architecture (Phase 2)
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

    # Phase 2 GP loop
    p["num_generations"] = _pick(args.num_generations, s3.get("num_generations"), DEFAULTS["num_generations"])
    p["patience"]        = _pick(args.patience, s3.get("patience"), DEFAULTS["patience"])

    # Phase 3
    cli_phase3_enabled = _str_to_bool(args.phase3_enabled)
    p["phase3_enabled"]          = _pick(cli_phase3_enabled, ph3.get("enabled"), DEFAULTS["phase3_enabled"])
    p["phase3_epochs"]           = _pick(args.phase3_epochs, ph3.get("epochs"), DEFAULTS["phase3_epochs"])
    p["phase3_batch_size"]       = _pick(args.phase3_batch_size, ph3.get("batch_size"), DEFAULTS["phase3_batch_size"])
    p["phase3_learning_rate"]    = _pick(args.phase3_learning_rate, ph3.get("learning_rate"), DEFAULTS["phase3_learning_rate"])
    p["phase3_patience"]         = _pick(args.phase3_patience, ph3.get("patience"), DEFAULTS["phase3_patience"])
    p["phase3_max_norm"]         = _pick(None, ph3.get("max_norm"), DEFAULTS["phase3_max_norm"])
    p["phase3_use_amp"]          = _pick(None, ph3.get("use_amp"), DEFAULTS["phase3_use_amp"])
    p["phase3_lambda_anchor"]    = _pick(args.phase3_lambda_anchor, ph3.get("lambda_anchor"), DEFAULTS["phase3_lambda_anchor"])
    p["phase3_pipeline_mode"]    = _pick(args.phase3_pipeline_mode, ph3.get("pipeline_mode"), DEFAULTS["phase3_pipeline_mode"])
    p["phase3_random_dup_train"] = _pick(None, ph3.get("random_dup_train"), DEFAULTS["phase3_random_dup_train"])

    p["no_save"]      = args.no_save
    p["force_phase1"] = args.force_phase1

    return p


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Dataset building (shared by Phase 1 and Phase 3)
# ──────────────────────────────────────────────────────────────────────

def _build_datasets(params, dataset_info):
    """Load CSVs + generate conformers + build datasets.

    Returns:
        train_ds (MolKConfDataset),
        valid_ds (IndexedMolKConfDataset),
        test_ds  (IndexedMolKConfDataset),
        task_type (str), num_tasks (int),
        device (torch.device).
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

    return train_ds, valid_ds, test_ds, task_type, num_tasks, device


# ──────────────────────────────────────────────────────────────────────
# Phase 1 orchestration
# ──────────────────────────────────────────────────────────────────────

def run_phase1(params, dataset_info, output_dir,
               train_ds, valid_ds, test_ds,
               task_type, num_tasks, device):
    """Run Phase 1: train UniMol with K confs, extract embeddings cache.

    Returns: (cache_path, phase1_results_dict)
    """
    dataset = dataset_info["name"]
    split_seed = params["split_seed"]
    K = params["K"]

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


# ──────────────────────────────────────────────────────────────────────
# Phase 3 orchestration
# ──────────────────────────────────────────────────────────────────────

def run_phase3(params, dataset_info, output_dir, cache_path,
               train_ds, valid_ds, test_ds,
               task_type, num_tasks, device,
               phase2_results, phase1_model_path):
    """Run Phase 3: encoder fine-tune with frozen GP+ridge readout.

    Returns: phase3_results dict (or None on error/skipped).
    """
    from src.trainers.step3_phase3_train import Phase3Config, Phase3Trainer

    if task_type != "regression":
        print(f"\n[Phase 3 skipped: task_type={task_type} (regression-only)]")
        return None

    print("\n" + "=" * 70)
    print("Phase 3: Encoder fine-tuning")
    print("=" * 70)

    # Locate Phase 2 best individual
    best_individual_path = os.path.join(output_dir, "best_individual.pt")
    if not os.path.exists(best_individual_path):
        print(f"\n✗ Phase 3 skipped: best_individual.pt not found at {best_individual_path}")
        return None

    if not os.path.exists(phase1_model_path):
        print(f"\n✗ Phase 3 skipped: Phase 1 model not found at {phase1_model_path}")
        return None

    ph3_cfg = Phase3Config(
        K=params["K"],
        epochs=params["phase3_epochs"],
        batch_size=params["phase3_batch_size"],
        learning_rate=params["phase3_learning_rate"],
        patience=params["phase3_patience"],
        max_norm=params["phase3_max_norm"],
        use_amp=params["phase3_use_amp"],
        lambda_anchor=params["phase3_lambda_anchor"],
        pipeline_mode=params["phase3_pipeline_mode"],
        random_dup_train=params["phase3_random_dup_train"],
        training_seed=params["random_seed"],
        fixed_seed_eval=0,
    )

    phase3_output_dir = os.path.join(output_dir, "phase3")
    os.makedirs(phase3_output_dir, exist_ok=True)

    trainer = Phase3Trainer(
        cfg=ph3_cfg,
        task_type=task_type,
        num_tasks=num_tasks,
        device=device,
        phase1_model_path=phase1_model_path,
        best_individual_path=best_individual_path,
        cache_path=cache_path,
        gp_input_dim=params["D"],
        baseline_phase2_valid_mse=phase2_results["valid_mse"],
        baseline_phase2_test_mse=phase2_results["test_mse"],
    )
    return trainer.run(train_ds, valid_ds, test_ds, output_dir=phase3_output_dir)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Phase 1 (UniMol multi-conf) + Phase 2 (GP) + Phase 3 (encoder fine-tune)",
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

    # Phase 3
    parser.add_argument("--phase3-enabled", type=str, default=None,
                        help="true/false (default from config or true)")
    parser.add_argument("--phase3-epochs", type=int, default=None)
    parser.add_argument("--phase3-batch-size", type=int, default=None)
    parser.add_argument("--phase3-learning-rate", type=float, default=None)
    parser.add_argument("--phase3-patience", type=int, default=None)
    parser.add_argument("--phase3-lambda-anchor", type=float, default=None)
    parser.add_argument("--phase3-pipeline-mode", type=str, default=None,
                        choices=["safe", "vanilla"])

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
    print("UniMol-GP -- Step 3: Phase 1 (UniMol Multi-Conf) + Phase 2 (GP) + Phase 3 (fine-tune)")
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
    print(f"--- Phase 3 (encoder fine-tune) ---")
    print(f"enabled                 : {params['phase3_enabled']}")
    if params["phase3_enabled"]:
        print(f"epochs                  : {params['phase3_epochs']}")
        print(f"batch_size              : {params['phase3_batch_size']}")
        print(f"learning_rate           : {params['phase3_learning_rate']}")
        print(f"patience                : {params['phase3_patience']}")
        print(f"λ_anchor                : {params['phase3_lambda_anchor']}")
        print(f"pipeline_mode           : {params['phase3_pipeline_mode']}")
    if not params["no_save"]:
        print(f"Save to                 : {OUTPUT_DIR}/step3/{args.dataset}/seed_{split_seed}/{timestamp}/")
    print("=" * 70)

    # ── Scope guard: regression only (Phase 2 GP) ────────────────
    if dataset_info["task_type"] != "regression":
        print(f"\n✗ Error: Phase 2 GP scope is regression only, got "
              f"task_type={dataset_info['task_type']} for dataset {args.dataset}")
        print("   (Phase 1 supports all task types, but Phase 2 hasn't been extended yet.)")
        sys.exit(1)

    # ── Cache check ──────────────────────────────────────────────
    K = params["K"]
    cache_dir = params["cache_dir"]
    have_cache = cache_exists(cache_dir, args.dataset, split_seed, K)

    output_dir = os.path.join(OUTPUT_DIR, "step3", args.dataset,
                               f"seed_{split_seed}", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # ── Decide whether to build datasets ─────────────────────────
    # Datasets are needed for:
    #   - Phase 1 run (cache miss or --force-phase1)
    #   - Phase 3 run (regardless of cache, if enabled & regression)
    need_datasets = (
        (not have_cache) or params["force_phase1"]
        or (params["phase3_enabled"] and dataset_info["task_type"] == "regression")
    )

    train_ds = valid_ds = test_ds = None
    task_type = dataset_info["task_type"]
    num_tasks = 1
    use_gpu = params["use_gpu"] and torch.cuda.is_available()
    device = torch.device(f"cuda:{params['gpu_id']}") if use_gpu else torch.device("cpu")

    if need_datasets:
        print("\n[Building datasets — needed for Phase 1 run and/or Phase 3]")
        try:
            train_ds, valid_ds, test_ds, task_type, num_tasks, device = _build_datasets(
                params, dataset_info,
            )
        except Exception as e:
            print(f"\n✗ Dataset build error: {type(e).__name__}: {e}")
            raise
    else:
        print(f"\n[Datasets not built — using cache only, Phase 3 disabled or non-regression]")

    # ── Phase 1: cache check + run ───────────────────────────────
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
            cache_path, phase1_results = run_phase1(
                params, dataset_info, output_dir,
                train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
                task_type=task_type, num_tasks=num_tasks, device=device,
            )
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

    # ── Phase 3: encoder fine-tune (optional) ────────────────────
    phase3_results = None
    if params["phase3_enabled"]:
        # Locate Phase 1 model path
        if phase1_results is not None:
            phase1_model_path = phase1_results["model_path"]
        else:
            # Phase 1 was cached/skipped — find from cache metadata or default location
            try:
                from src.data.embeddings_cache import load_cache
                _cache = load_cache(cache_path, map_location='cpu')
                phase1_model_path = _cache["metadata"].get("phase1_model_path")
            except Exception:
                phase1_model_path = None
            if not phase1_model_path or not os.path.exists(phase1_model_path):
                # Fall back: search for it in current output_dir or previous timestamps
                phase1_model_path = os.path.join(output_dir, "phase1", "phase1_model.pth")

        try:
            phase3_results = run_phase3(
                params, dataset_info, output_dir, cache_path,
                train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
                task_type=task_type, num_tasks=num_tasks, device=device,
                phase2_results=results, phase1_model_path=phase1_model_path,
            )
        except KeyboardInterrupt:
            print("\n[Phase 3 interrupted by user]")
            phase3_results = {"chosen_model": "phase2", "interrupted": True}
        except Exception as e:
            print(f"\n✗ Phase 3 error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            phase3_results = {"chosen_model": "phase2", "error": str(e)}

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
    print(f"Test  RMSE     : {results['test_rmse']:.4f}  (MSE={results['test_mse']:.4f})")
    print(f"Total time     : {results['total_time_s']:.1f}s "
          f"({results['num_generations_run']} generations)")

    if phase3_results is not None and "phase3_metrics" in phase3_results:
        m3 = phase3_results["phase3_metrics"]
        print(f"--- Phase 3 (encoder fine-tune) ---")
        for split in ("train", "valid", "test"):
            print(f"{split.capitalize():5s} | "
                  f"MSE={m3[split]['mse']:.4f}, RMSE={m3[split]['rmse']:.4f}")
        chosen = phase3_results["chosen_model"]
        save_strategy = phase3_results.get("save_strategy", "always_phase3")
        print(f"Save strategy  : {save_strategy}")
        print(f"Chosen model   : {chosen.upper()}")
        # Diagnostic deltas (Phase 3 - Phase 2)
        d_valid = m3['valid']['mse'] - results['valid_mse']
        d_test  = m3['test']['mse']  - results['test_mse']
        v_arrow = "↓ better" if d_valid < 0 else "↑ worse"
        t_arrow = "↓ better" if d_test  < 0 else "↑ worse"
        print(f"Δvalid MSE     : {d_valid:+.4f}  ({v_arrow})")
        print(f"Δtest  MSE     : {d_test:+.4f}  ({t_arrow})")

    # Final answer — Phase 3 is always canonical when it ran successfully
    print(f"\n--- CANONICAL RESULT ---")
    if phase3_results is not None and "phase3_metrics" in phase3_results:
        m3 = phase3_results["phase3_metrics"]
        print(f"Test  RMSE     : {m3['test']['rmse']:.4f}  ← MAIN (Phase 3)")
    else:
        # Phase 3 disabled, errored, or interrupted — fall back to Phase 2
        reason = "disabled"
        if phase3_results is not None:
            if "error" in phase3_results:
                reason = f"errored: {phase3_results['error'][:60]}"
            elif "interrupted" in phase3_results:
                reason = "interrupted"
        print(f"Test  RMSE     : {results['test_rmse']:.4f}  "
              f"← MAIN (Phase 2 fallback — Phase 3 {reason})")

    if not params["no_save"]:
        print(f"\nSaved to       : {OUTPUT_DIR}/{experiment_name}/")
    print("=" * 70)
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
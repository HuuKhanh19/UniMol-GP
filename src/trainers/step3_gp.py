"""
Step 3: UniMol (frozen from step 1) + Multi-tree GP + double ridge readout.

Training loop:
    Per generation g:
        1. Resample train with random duplication (per-gen seeded)
        2. Compute Z_train features via K Forests (pop=N super-individuals)
        3. Fit ridges on train → evaluate train fitness AND valid fitness batched
        4. best_idx_valid = argmax(valid fitness)
        5. If valid improved → snapshot best state (trees + ridges + metrics)
        6. evolve_step(train fitness)
        7. Early stop if no improvement for `patience` gens

Final:
    - Report best-of-run: train/valid/test MSE, gen found
    - Save results.json + best_individual.pt (trees + ridges)
    - Dump sympy formulas to formulas.txt
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..data.datasets import OUTPUT_DIR
from ..data.embeddings_cache import (
    fixed_K_sample,
    load_cache,
    reduce_and_normalize_splits,
    sample_K_with_duplication,
)
from ..models.gp_model import GPConfig, MultiTreeGPModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Best-of-run snapshot
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BestSnapshot:
    gen: int                                      # gen khi snapshot
    individual_idx: int                           # idx trong pop tại gen đó (informational)
    train_mse: float                              # of best individual on current train resample
    valid_mse: float                              # on fixed valid (selection criterion)
    test_mse: float                               # on fixed test (computed at snapshot time)
    trees_state: list                             # from model.snapshot_individual_trees(idx)
    ridge_params: Dict[str, torch.Tensor]         # from model.get_individual_ridges(...)
    sympy_formulas: list                          # from model.get_sympy_formulas(idx)

    def to_serializable(self) -> Dict[str, Any]:
        """Dict dùng để torch.save (tensors kept as CPU tensors)."""
        return {
            "gen": self.gen,
            "individual_idx": self.individual_idx,
            "train_mse": self.train_mse,
            "valid_mse": self.valid_mse,
            "test_mse": self.test_mse,
            "trees_state": self.trees_state,
            "ridge_params": self.ridge_params,
            "sympy_formulas": [str(f) for f in self.sympy_formulas],
        }


# ──────────────────────────────────────────────────────────────────────
# Step3Trainer
# ──────────────────────────────────────────────────────────────────────

class Step3Trainer:
    """
    Trains K×q multi-tree GP + 2-level ridge on precomputed UniMol embeddings.
    """

    def __init__(self, params: dict, dataset_info: dict, experiment_name: str):
        self.p = params
        self.dataset_info = dataset_info
        self.task_type = dataset_info["task_type"]

        if self.task_type != "regression":
            raise ValueError(
                f"Step 3 scope is regression only, got task_type={self.task_type}"
            )

        self.output_dir = os.path.join(OUTPUT_DIR, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Device
        gpu_id = params.get("gpu_id", 0)
        use_gpu = params.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        # Seeds
        torch.manual_seed(params["random_seed"])
        np.random.seed(params["random_seed"])

        # Build GP config
        self.gp_cfg = GPConfig(
            K=params["K"],
            num_trees_per_conformer=params["num_trees_per_conformer"],
            D=params["D"],
            pop_size=params["pop_size"],
            max_tree_len=params["max_tree_len"],
            max_layer_cnt=params["max_layer_cnt"],
            mutation_max_layer_cnt=params["mutation_max_layer_cnt"],
            using_funcs=tuple(params["using_funcs"]),
            const_samples=tuple(params["const_samples"]),
            mutation_rate=params["mutation_rate"],
            survival_rate=params["survival_rate"],
            elite_rate=params["elite_rate"],
            parsimony_alpha=params["parsimony_alpha"],
            lambda_inner=params["lambda_inner"],
            lambda_outer=params["lambda_outer"],
        )

    # ─────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────

    def _load_data(self, cache_path: str) -> dict:
        """Load cache, apply PCA reduction, fixed-sample valid/test."""
        cache = load_cache(cache_path, map_location="cpu")
        K = self.gp_cfg.K

        # Sanity checks on cache
        K_cache = cache["metadata"]["K_target"]
        if K_cache != K:
            raise ValueError(
                f"Cache K={K_cache} mismatch with config K={K}. "
                f"Recompute cache with --force."
            )
        D_orig = cache["metadata"]["D"]
        logger.info(f"Cache D_orig={D_orig}, applying PCA → {self.gp_cfg.D}")

        # ── PCA reduction: 512 → gp_input_dim ──
        # Workaround cho evogp CUDA stack limit (varLen <= MAX_STACK / 4)
        reduced, self.transform_info = reduce_and_normalize_splits(
            cache, gp_input_dim=self.gp_cfg.D
        )
        cache = reduced  # use reduced version from now on

        # Fixed sample for valid/test (seed=0)
        X_valid = fixed_K_sample(
            cache["valid"]["X"], cache["valid"]["valid_counts"], K=K, seed=0
        ).to(self.device)
        X_test = fixed_K_sample(
            cache["test"]["X"], cache["test"]["valid_counts"], K=K, seed=0
        ).to(self.device)

        # Train stays as raw cache tensors (to resample per gen)
        data = {
            "X_train_raw":    cache["train"]["X"],                # (N_tr, K, d_gp) CPU
            "vc_train":       cache["train"]["valid_counts"],     # (N_tr,) CPU
            "y_train":        cache["train"]["y"].to(self.device),
            "X_valid":        X_valid,                             # fixed, on device
            "y_valid":        cache["valid"]["y"].to(self.device),
            "X_test":         X_test,                              # fixed, on device
            "y_test":         cache["test"]["y"].to(self.device),
            "metadata":       cache["metadata"],
        }

        logger.info(
            f"Data loaded | train={len(data['y_train'])} "
            f"valid={len(data['y_valid'])} test={len(data['y_test'])} "
            f"K={K} D_gp={self.gp_cfg.D} "
            f"(PCA explained var: {self.transform_info['pca_explained_var']:.2%})"
        )
        return data

    # ─────────────────────────────────────────────────────────────
    # Train loop
    # ─────────────────────────────────────────────────────────────

    def run(self, cache_path: str) -> Dict[str, Any]:
        """Main entry. Returns results dict."""
        print(f"\n{'='*60}")
        print(f"Step 3: GP + Ridge -- {self.dataset_info['name']}")
        print(f"Task: {self.task_type} | Device: {self.device}")
        print(f"K={self.gp_cfg.K}, num_trees_per_conformer={self.gp_cfg.num_trees_per_conformer}, "
              f"pop={self.gp_cfg.pop_size}, "
              f"max_tree_len={self.gp_cfg.max_tree_len}, max_layer_cnt={self.gp_cfg.max_layer_cnt}")
        print(f"{'='*60}\n")

        # Load data
        data = self._load_data(cache_path)

        # Build model
        logger.info("Building MultiTreeGPModel ...")
        model = MultiTreeGPModel(self.gp_cfg, device=self.device)

        # Train loop
        best: Optional[BestSnapshot] = None
        patience_counter = 0

        num_generations = self.p["num_generations"]
        patience = self.p["patience"]

        # Per-gen seed for conformer resampling (reproducible)
        resample_gen = torch.Generator(device=self.device)

        # Train stats log
        history = []

        t_start = time.time()
        for gen in range(num_generations):
            t_gen = time.time()

            # ── 1. Resample train with per-gen seed ──────────────────
            resample_gen.manual_seed(self.p["random_seed"] * 10000 + gen)
            X_train = sample_K_with_duplication(
                data["X_train_raw"].to(self.device),
                data["vc_train"].to(self.device),
                K=self.gp_cfg.K,
                generator=resample_gen,
            )  # (N_tr, K, D) on device

            # ── 2-3. Efficient dual eval: fit ridges on train, eval both ─
            fit_fitness, valid_fitness, ridges = model.evaluate_dual(
                X_fit=X_train,
                y_fit=data["y_train"],
                X_eval=data["X_valid"],
                y_eval=data["y_valid"],
            )
            # fit_fitness: (pop,) = -train_MSE per super-individual
            # valid_fitness: (pop,) = -valid_MSE per super-individual
            # ridges: batched dict (ridges fit on train resample)

            best_train_fitness = fit_fitness.max().item()
            best_valid_fitness = valid_fitness.max().item()
            best_train_mse = -best_train_fitness
            best_valid_mse = -best_valid_fitness

            # ── 4. Best super-individual BY VALID MSE ────────────────
            best_idx_valid = valid_fitness.argmax().item()

            # ── 5. If improved → snapshot ────────────────────────────
            improved = (best is None) or (best_valid_mse < best.valid_mse - 1e-8)
            if improved:
                # Compute test MSE for this individual using same ridges
                y_pred_test_all = model.predict_with_ridges(data["X_test"], ridges)
                err = y_pred_test_all[best_idx_valid] - data["y_test"]
                err = torch.nan_to_num(err, nan=1e6, posinf=1e6, neginf=1e6)
                test_mse_best = (err ** 2).mean().item()

                # Get individual's train MSE (using same fit)
                train_mse_this_ind = -fit_fitness[best_idx_valid].item()

                # Snapshot trees and ridges
                trees_state = model.snapshot_individual_trees(best_idx_valid)
                ridge_params = model.get_individual_ridges(ridges, best_idx_valid)

                # Sympy formulas (can be slow; only compute when improved)
                try:
                    sympy_formulas = model.get_sympy_formulas(best_idx_valid)
                except Exception as e:
                    sympy_formulas = [f"<error computing sympy: {e}>"]

                best = BestSnapshot(
                    gen=gen,
                    individual_idx=best_idx_valid,
                    train_mse=train_mse_this_ind,
                    valid_mse=best_valid_mse,
                    test_mse=test_mse_best,
                    trees_state=trees_state,
                    ridge_params=ridge_params,
                    sympy_formulas=sympy_formulas,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Log
            dt = time.time() - t_gen
            marker = "★" if improved else " "
            logger.info(
                f"{marker} Gen {gen:3d}  "
                f"pop_best_train_mse={best_train_mse:.4f}  "
                f"pop_best_valid_mse={best_valid_mse:.4f}  "
                f"best-of-run_valid_mse={best.valid_mse:.4f} "
                f"(test={best.test_mse:.4f}, gen={best.gen})  "
                f"[{dt:.1f}s]"
            )
            history.append({
                "gen": gen,
                "train_mse": best_train_mse,
                "valid_mse": best_valid_mse,
                "best_valid_mse": best.valid_mse,
                "best_test_mse": best.test_mse,
                "best_gen": best.gen,
                "improved": improved,
                "elapsed_s": dt,
            })

            # ── 7. Early stop ─────────────────────────────────────────
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping at gen {gen} "
                    f"(patience={patience} without improvement)"
                )
                break

            # ── 6. Evolve all K Forests with TRAIN fitness ───────────
            model.evolve_step(fit_fitness)

        total_time = time.time() - t_start
        logger.info(f"\nTotal training time: {total_time:.1f}s")

        # ── Final results ────────────────────────────────────────────
        assert best is not None, "No generations ran — check config"

        rmse = lambda mse: float(np.sqrt(mse))
        results = {
            "dataset":     self.dataset_info["name"],
            "task":        self.task_type,
            "best_gen":    best.gen,
            "best_individual_idx": best.individual_idx,
            "train_mse":   best.train_mse,
            "train_rmse":  rmse(best.train_mse),
            "valid_mse":   best.valid_mse,
            "valid_rmse":  rmse(best.valid_mse),
            "test_mse":    best.test_mse,
            "test_rmse":   rmse(best.test_mse),
            "total_time_s":    total_time,
            "num_generations_run": len(history),
            "history":     history,
            "metadata":    data["metadata"],
        }

        # Save
        self._save_outputs(results, best)
        return results

    # ─────────────────────────────────────────────────────────────
    # Save artifacts
    # ─────────────────────────────────────────────────────────────

    def _save_outputs(self, results: dict, best: BestSnapshot) -> None:
        if self.p.get("no_save", False):
            return

        # results.json (metrics + history)
        res_path = os.path.join(self.output_dir, "results.json")

        def _json_safe(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, torch.Tensor):
                return o.tolist()
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_json_safe(x) for x in o]
            return o

        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(results), f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Results saved → {res_path}")

        # best_individual.pt
        best_path = os.path.join(self.output_dir, "best_individual.pt")
        torch.save({
            "config": asdict(self.gp_cfg),
            "snapshot": best.to_serializable(),
        }, best_path)
        logger.info(f"Best individual saved → {best_path}")

        # formulas.txt
        formulas_path = os.path.join(self.output_dir, "formulas.txt")
        with open(formulas_path, "w", encoding="utf-8") as f:
            f.write(f"Best super-individual formulas\n")
            f.write(f"Gen: {best.gen} | individual_idx: {best.individual_idx}\n")
            f.write(f"Train MSE: {best.train_mse:.6f} | Valid MSE: {best.valid_mse:.6f} | "
                    f"Test MSE: {best.test_mse:.6f}\n")
            f.write(f"{'=' * 60}\n\n")

            w_inner = best.ridge_params["w_inner"]  # (K, q)
            b_inner = best.ridge_params["b_inner"]  # (K,)
            w_outer = best.ridge_params["w_outer"]  # (K,)
            b_outer = best.ridge_params["b_outer"]  # scalar

            f.write(f"Outer ridge (K={self.gp_cfg.K} conformers + bias):\n")
            f.write(f"  w_outer = {w_outer.tolist()}\n")
            f.write(f"  b_outer = {float(b_outer):.6f}\n\n")

            for k in range(self.gp_cfg.K):
                f.write(f"{'─' * 60}\n")
                f.write(f"Conformer slot k={k}\n")
                f.write(f"  Inner ridge: w_inner[{k}] = {w_inner[k].tolist()}\n")
                f.write(f"               b_inner[{k}] = {float(b_inner[k]):.6f}\n")
                f.write(f"  GP trees (num_trees_per_conformer={self.gp_cfg.num_trees_per_conformer}):\n")
                exprs = best.sympy_formulas[k]
                if isinstance(exprs, list):
                    for j, expr in enumerate(exprs):
                        f.write(f"    f_{j}(x)  = {expr}\n")
                else:
                    f.write(f"    {exprs}\n")
                f.write("\n")
        logger.info(f"Formulas saved → {formulas_path}")
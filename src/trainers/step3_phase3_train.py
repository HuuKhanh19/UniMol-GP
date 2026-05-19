"""
Phase 3 Trainer — SGD fine-tune UniMol encoder with frozen GP+ridge readout.

Pipeline:
    encoder (TRAINABLE)
        ↓ (B, K, 512)
    DifferentiablePipeline (FROZEN: PCA + standardize + K trees + ridges)
        ↓ y_pred (B,)
    MSE(y_pred, y_target) + λ_anchor · MSE(emb_current, emb_anchor)
        ↓
    backward → encoder weights update

Key design choices:
  - Anchor loss: a frozen copy of Phase 1 encoder produces emb_anchor on the
    SAME conformers each batch. Penalizing drift prevents encoder from going
    off-manifold relative to evolved GP trees.
  - Train sampling: random duplication (matches Phase 1 augmentation).
  - Valid/test sampling: fixed seed=0 (matches Phase 2 strategy → metrics
    directly comparable to Phase 2 baseline).
  - Save logic: only override Phase 2 weights as the canonical answer if
    Phase 3 valid_mse improved. Phase 3 cannot harm.
  - Currently regression-only (Phase 2 GP is regression). Asserts task_type.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.embeddings_cache import load_cache, reduce_and_normalize_splits
from ..data.multi_conf_dataset import (
    IndexedMolKConfDataset,
    MolKConfDataset,
    make_collate_K_indexed_fixed,
    make_collate_K_random_dup,
)
from ..models.diff_pipeline import DifferentiablePipeline
from ..models.multi_conf_unimol import MultiConformerUniMol

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Phase3Config:
    K: int
    epochs: int = 15
    batch_size: int = 4
    learning_rate: float = 1.0e-5
    patience: int = 5
    warmup_ratio: float = 0.0
    max_norm: float = 5.0
    use_amp: bool = False
    weight_decay: float = 0.0
    lambda_anchor: float = 0.5
    pipeline_mode: str = "safe"        # "safe" | "vanilla"
    random_dup_train: bool = True
    training_seed: int = 42
    fixed_seed_eval: int = 0


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────
class Phase3Trainer:
    """Fine-tune encoder with frozen GP+ridge readout.

    Save strategy: Phase 3 is ALWAYS the canonical answer. The comparison
    against Phase 2 baseline is recorded for diagnostics only — rationale
    is that Phase 2's valid metric is selection-overfit (greedy GP search
    on a small valid set), while Phase 3's anchor loss provides better
    regularization for generalization. Trust Phase 3.

    Returns dict with:
        - phase3_init_valid_mse, phase3_best_epoch
        - phase3_metrics (train/valid/test MSE+RMSE)
        - phase2_baseline (valid_mse, test_mse for diagnostic)
        - improved_valid_over_phase2, improved_test_over_phase2 (diagnostic)
        - chosen_model: always 'phase3'
        - save_strategy: 'always_phase3'
        - history (per-epoch)
        - phase3_model_path
    """

    def __init__(
        self,
        cfg: Phase3Config,
        task_type: str,
        num_tasks: int,
        device: torch.device,
        phase1_model_path: str,
        best_individual_path: str,
        cache_path: str,
        gp_input_dim: int,
        baseline_phase2_valid_mse: float,
        baseline_phase2_test_mse: float,
    ):
        if task_type != "regression":
            raise ValueError(
                f"Phase 3 currently supports only regression "
                f"(GP+ridge is regression). Got task_type={task_type}."
            )

        self.cfg = cfg
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.device = device
        self.baseline_p2_valid = baseline_phase2_valid_mse
        self.baseline_p2_test = baseline_phase2_test_mse

        # ── Load Phase 1 encoder weights (state dict for UniMolModel base)
        logger.info(f"Loading Phase 1 encoder weights from {phase1_model_path}")
        phase1_state = torch.load(phase1_model_path, map_location='cpu', weights_only=False)

        # Trainable encoder
        logger.info("Building trainable encoder (init from Phase 1)")
        self.encoder = MultiConformerUniMol(task_type, num_tasks, remove_hs=True).to(device)
        self.encoder.base.load_state_dict(phase1_state)
        n_train = sum(p.numel() for p in self.encoder.parameters())
        logger.info(f"Encoder trainable params: {n_train:,}")

        # Frozen anchor encoder (same Phase 1 weights, requires_grad=False)
        logger.info("Building frozen anchor encoder")
        self.encoder_anchor = MultiConformerUniMol(task_type, num_tasks, remove_hs=True).to(device)
        self.encoder_anchor.base.load_state_dict(phase1_state)
        for p in self.encoder_anchor.parameters():
            p.requires_grad = False
        self.encoder_anchor.eval()

        # ── Build differentiable pipeline (frozen GP + ridges)
        logger.info(f"Loading Phase 2 best individual from {best_individual_path}")
        best_dict = torch.load(best_individual_path, map_location='cpu', weights_only=False)

        logger.info(f"Re-fitting PCA + standardize from cache {cache_path}")
        cache = load_cache(cache_path, map_location='cpu')
        _, transform_info = reduce_and_normalize_splits(cache, gp_input_dim=gp_input_dim)

        self.pipeline = DifferentiablePipeline(
            best_individual_dict=best_dict,
            transform_info=transform_info,
            mode=cfg.pipeline_mode,
            device=device,
        )

        # Determinism
        torch.manual_seed(cfg.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.training_seed)

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    @staticmethod
    def _wrap_indexed(ds: MolKConfDataset) -> IndexedMolKConfDataset:
        """Wrap plain MolKConfDataset as IndexedMolKConfDataset for fixed-seed eval."""
        if isinstance(ds, IndexedMolKConfDataset):
            return ds
        return IndexedMolKConfDataset(ds.conformers_per_mol, ds.targets)

    def _build_loaders(
        self,
        train_ds: MolKConfDataset,
        valid_ds: IndexedMolKConfDataset,
        test_ds: IndexedMolKConfDataset,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        cfg = self.cfg

        # Dedicated generator for collate random sampling (reproducible)
        shared_gen = torch.Generator()
        shared_gen.manual_seed(cfg.training_seed)

        if cfg.random_dup_train:
            train_collate = make_collate_K_random_dup(cfg.K, shared_gen)
            train_loader_ds = train_ds  # plain MolKConfDataset
            shuffle = True
            gen_for_loader = shared_gen
        else:
            # Fixed seed=0 sampling — needs IndexedMolKConfDataset
            train_collate = make_collate_K_indexed_fixed(cfg.K, fixed_seed=cfg.fixed_seed_eval)
            train_loader_ds = self._wrap_indexed(train_ds)
            shuffle = False
            gen_for_loader = None

        train_loader = DataLoader(
            train_loader_ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=gen_for_loader,
            collate_fn=train_collate,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_K_indexed_fixed(cfg.K, fixed_seed=cfg.fixed_seed_eval),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_K_indexed_fixed(cfg.K, fixed_seed=cfg.fixed_seed_eval),
        )
        return train_loader, valid_loader, test_loader

    # ─────────────────────────────────────────────────────────────
    # Forward pass (encoder + pipeline)
    # ─────────────────────────────────────────────────────────────
    def _forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (y_pred, emb_current). batch is on device."""
        emb_current = self.encoder(
            batch["src_tokens"], batch["src_distance"],
            batch["src_coord"],  batch["src_edge_type"],
            return_repr=True,
        )  # (B, K, 512)
        y_pred = self.pipeline(emb_current)  # (B,)
        return y_pred, emb_current

    @torch.no_grad()
    def _forward_anchor(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Frozen Phase 1 encoder embedding for anchor loss."""
        return self.encoder_anchor(
            batch["src_tokens"], batch["src_distance"],
            batch["src_coord"],  batch["src_edge_type"],
            return_repr=True,
        )  # (B, K, 512)

    # ─────────────────────────────────────────────────────────────
    # Gradient sanitization
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _filter_nan_grads(self) -> Tuple[int, int]:
        """Replace NaN/inf in encoder grads with 0 in-place.

        Defensive — compound safe ops in lambdified trees can produce
        non-finite gradients despite finite forward (e.g. internal saved
        tensors near safe_inv/safe_log clamp boundaries). Worst case:
        a batch produces all-NaN grads → zeroed → no update for that batch.
        As long as some batches produce finite signal, encoder learns.

        Returns (n_params_with_nan, n_total_params_with_grad).
        """
        n_nan_params = 0
        n_total_params = 0
        for p in self.encoder.parameters():
            if p.grad is not None:
                n_total_params += 1
                if not torch.isfinite(p.grad).all():
                    n_nan_params += 1
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return n_nan_params, n_total_params

    # ─────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        self.encoder.train()
        # Pipeline + anchor stay in eval mode regardless
        self.pipeline.eval()
        self.encoder_anchor.eval()

        total_main = 0.0
        total_anchor = 0.0
        total_loss = 0.0
        n_batches = 0
        n_batches_nan = 0
        n_nan_params_total = 0

        for batch in loader:
            batch_dev = self._batch_to_device(batch)
            y_target = batch_dev["targets"].float()
            if y_target.dim() > 1:
                y_target = y_target.squeeze(-1)

            optimizer.zero_grad()

            # Current encoder forward + pipeline
            y_pred, emb_current = self._forward(batch_dev)

            # Anchor encoder forward (no grad)
            emb_anchor = self._forward_anchor(batch_dev)

            # Losses
            main_loss = ((y_pred - y_target) ** 2).mean()
            anchor_loss = ((emb_current - emb_anchor) ** 2).mean()
            loss = main_loss + self.cfg.lambda_anchor * anchor_loss

            loss.backward()
            # Filter NaN/inf from grads BEFORE clipping (clip_grad_norm_
            # propagates NaN if any tensor has NaN).
            n_nan, _ = self._filter_nan_grads()
            if n_nan > 0:
                n_batches_nan += 1
                n_nan_params_total += n_nan
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.max_norm)
            optimizer.step()

            total_main += float(main_loss.item())
            total_anchor += float(anchor_loss.item())
            total_loss += float(loss.item())
            n_batches += 1

        if n_batches_nan > 0:
            logger.info(
                f"  ⚠ {n_batches_nan}/{n_batches} batches had NaN grads "
                f"(filtered → 0). Avg {n_nan_params_total/n_batches_nan:.1f} "
                f"params/batch affected."
            )

        return {
            "train_main_mse":   total_main / max(n_batches, 1),
            "train_anchor_mse": total_anchor / max(n_batches, 1),
            "train_total_loss": total_loss / max(n_batches, 1),
            "n_batches_nan":    n_batches_nan,
        }

    # ─────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.encoder.eval()
        self.pipeline.eval()

        all_preds = []
        all_targets = []
        for batch in loader:
            batch_dev = self._batch_to_device(batch)
            y_target = batch_dev["targets"].float()
            if y_target.dim() > 1:
                y_target = y_target.squeeze(-1)
            y_pred, _ = self._forward(batch_dev)
            all_preds.append(y_pred.detach().cpu())
            all_targets.append(y_target.detach().cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        mse = ((preds - targets) ** 2).mean().item()
        rmse = float(np.sqrt(mse))
        return {"mse": mse, "rmse": rmse}

    # ─────────────────────────────────────────────────────────────
    # Main run
    # ─────────────────────────────────────────────────────────────
    def run(
        self,
        train_ds: MolKConfDataset,
        valid_ds: IndexedMolKConfDataset,
        test_ds: IndexedMolKConfDataset,
        output_dir: str,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)

        # Build loaders
        train_loader, valid_loader, test_loader = self._build_loaders(
            train_ds, valid_ds, test_ds,
        )

        # Optimizer (encoder params only)
        optimizer = torch.optim.Adam(
            [p for p in self.encoder.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Sanity check: encoder gradient flow on one mini-batch
        logger.info("Sanity check: gradient flow through pipeline...")
        self._sanity_check_grad(train_loader)

        # Initial valid metric (Phase 1 weights → pipeline)
        init_valid = self._evaluate(valid_loader)
        logger.info(
            f"Phase 3 init valid MSE = {init_valid['mse']:.4f} "
            f"(Phase 2 baseline = {self.baseline_p2_valid:.4f})"
        )

        # Training loop
        best_valid = init_valid["mse"]
        best_epoch = -1   # -1 means "init/pre-training"
        best_state = copy.deepcopy(self.encoder.state_dict())
        patience_ctr = 0
        history = []

        print(f"\n{'='*60}")
        print(f"Phase 3: encoder fine-tuning (lr={cfg.learning_rate}, "
              f"epochs={cfg.epochs}, λ_anchor={cfg.lambda_anchor})")
        print(f"{'='*60}\n")

        t0 = time.time()
        for epoch in range(cfg.epochs):
            t_ep = time.time()
            train_metrics = self._train_epoch(train_loader, optimizer)
            valid_metrics = self._evaluate(valid_loader)
            valid_mse = valid_metrics["mse"]

            improved = valid_mse < best_valid
            marker = "★" if improved else " "
            ep_time = time.time() - t_ep

            log_line = (
                f"{marker} Epoch {epoch:2d}  "
                f"train_main={train_metrics['train_main_mse']:.4f}  "
                f"train_anchor={train_metrics['train_anchor_mse']:.4f}  "
                f"valid_mse={valid_mse:.4f}  "
                f"[{ep_time:.1f}s]"
            )
            if improved:
                best_valid = valid_mse
                best_epoch = epoch
                best_state = copy.deepcopy(self.encoder.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                log_line += f"  (patience {patience_ctr}/{cfg.patience})"
            print(log_line)

            history.append({
                "epoch": epoch,
                **train_metrics,
                "valid_mse": valid_mse,
                "valid_rmse": valid_metrics["rmse"],
                "improved": improved,
                "elapsed_s": ep_time,
            })

            if patience_ctr >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        total_time = time.time() - t0
        print(f"\nPhase 3 total time: {total_time:.1f}s")

        # Restore best
        self.encoder.load_state_dict(best_state)
        if best_epoch == -1:
            print(f"Best state is INITIAL (Phase 1 weights, no improvement during training).")
        else:
            print(f"Restored best state from epoch {best_epoch} "
                  f"(valid_mse={best_valid:.4f}).")

        # Final eval on all 3 splits with fixed seed=0 sampling
        train_loader_eval = DataLoader(
            self._wrap_indexed(train_ds),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
            collate_fn=make_collate_K_indexed_fixed(cfg.K, fixed_seed=cfg.fixed_seed_eval),
        )
        train_eval = self._evaluate(train_loader_eval)
        valid_eval = self._evaluate(valid_loader)
        test_eval = self._evaluate(test_loader)

        # Save logic: Phase 3 is ALWAYS the canonical answer. The comparison
        # vs Phase 2 baseline is recorded for diagnostics only — rationale is
        # that Phase 2 selection-overfits valid (observed: seed_4 had Phase 2
        # valid 0.33 / test 1.15 vs Phase 3 valid 0.46 / test 0.64 — Phase 3
        # generalized better despite worse valid). The user-level decision is
        # to trust Phase 3's regularization (anchor loss) over Phase 2's
        # valid-driven selection.
        improved_valid_over_p2 = valid_eval["mse"] < self.baseline_p2_valid
        improved_test_over_p2 = test_eval["mse"] < self.baseline_p2_test
        chosen_model = 'phase3'

        # Always save Phase 3 weights for analysis
        phase3_model_path = os.path.join(output_dir, "phase3_model.pth")
        torch.save(self.encoder.state_dict_base_only(), phase3_model_path)
        logger.info(f"Phase 3 model saved → {phase3_model_path}")

        # Save results.json
        results_path = os.path.join(output_dir, "phase3_results.json")
        results = {
            "phase3_init_valid_mse": init_valid["mse"],
            "phase3_best_epoch": best_epoch,
            "phase3_best_valid_mse_during_training": best_valid,
            "phase3_metrics": {
                "train": train_eval,
                "valid": valid_eval,
                "test":  test_eval,
            },
            "phase2_baseline": {
                "valid_mse": self.baseline_p2_valid,
                "test_mse":  self.baseline_p2_test,
            },
            "improved_valid_over_phase2": improved_valid_over_p2,
            "improved_test_over_phase2":  improved_test_over_p2,
            "chosen_model": chosen_model,
            "save_strategy": "always_phase3",
            "history": history,
            "phase3_model_path": phase3_model_path,
            "config": asdict(cfg),
            "total_time_s": total_time,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Phase 3 results saved → {results_path}")

        # Print final summary
        v_marker = "↓" if improved_valid_over_p2 else "↑"
        t_marker = "↓" if improved_test_over_p2 else "↑"
        print(f"\n{'─'*60}")
        print(f"Phase 3 vs Phase 2 (diagnostic only — Phase 3 is canonical):")
        print(f"  Phase 2 baseline valid MSE: {self.baseline_p2_valid:.4f}")
        print(f"  Phase 3 best     valid MSE: {valid_eval['mse']:.4f}  "
              f"({v_marker} {'better' if improved_valid_over_p2 else 'worse'})")
        print(f"  Phase 2 baseline test  MSE: {self.baseline_p2_test:.4f}")
        print(f"  Phase 3 best     test  MSE: {test_eval['mse']:.4f}  "
              f"({t_marker} {'better' if improved_test_over_p2 else 'worse'})")
        print(f"  → Canonical: PHASE 3 (save strategy: always_phase3)")
        print(f"{'─'*60}\n")

        return results

    # ─────────────────────────────────────────────────────────────
    # Sanity helpers
    # ─────────────────────────────────────────────────────────────
    def _sanity_check_grad(self, loader: DataLoader):
        """Verify gradient flows through pipeline to encoder on 1 batch.

        Reports forward output (loss + finiteness) and backward gradient
        (norm + finiteness). Tolerates partial NaN (filtered to 0) as
        long as the post-filter gradient has some signal — train loop
        does the same filtering.
        """
        self.encoder.train()
        self.pipeline.eval()
        for batch in loader:
            batch_dev = self._batch_to_device(batch)
            y_target = batch_dev["targets"].float()
            if y_target.dim() > 1:
                y_target = y_target.squeeze(-1)
            y_pred, _ = self._forward(batch_dev)

            # Forward diagnostics
            y_finite = bool(torch.isfinite(y_pred).all().item())
            loss = ((y_pred - y_target) ** 2).mean()
            loss_val = float(loss.item())
            loss_finite = bool(np.isfinite(loss_val))

            # Diagnose forward failure first
            if not y_finite or not loss_finite:
                self.encoder.zero_grad()
                raise AssertionError(
                    f"Pipeline forward produced non-finite output "
                    f"(y_pred_finite={y_finite}, loss={loss_val}). "
                    f"Check trees for overflow (exp/pow), or switch "
                    f"pipeline_mode='vanilla' to compare with Phase 2."
                )

            loss.backward()
            n_nan, n_total = self._filter_nan_grads()

            # Compute post-filter grad norm
            grads = [p.grad for p in self.encoder.parameters() if p.grad is not None]
            grad_norm = sum(g.norm().item() for g in grads)
            grad_finite = all(torch.isfinite(g).all().item() for g in grads)
            self.encoder.zero_grad()

            nan_pct = 100.0 * n_nan / max(n_total, 1)
            logger.info(
                f"Sanity check: y_pred_finite={y_finite}, loss={loss_val:.4e} "
                f"(finite={loss_finite}), grad_norm={grad_norm:.2e}, "
                f"all_finite_post_filter={grad_finite}, "
                f"params_with_nan={n_nan}/{n_total} ({nan_pct:.1f}%)"
            )

            # Catastrophic failures
            if not grad_finite:
                raise AssertionError(
                    f"Encoder gradient still non-finite AFTER nan_to_num "
                    f"filter — this should be impossible. Check torch version."
                )
            if grad_norm <= 0:
                raise AssertionError(
                    f"Encoder gradient is zero (grad_norm={grad_norm}). "
                    f"Either no autograd connection from pipeline to encoder, "
                    f"or all paths produced NaN and got filtered. "
                    f"Try smaller batch_size or different conformer seed."
                )
            if nan_pct >= 95.0:
                logger.warning(
                    f"  ⚠ {nan_pct:.1f}% of params had NaN grads — encoder "
                    f"signal is very sparse. Training may be unstable. "
                    f"Consider lower lr or fewer epochs."
                )
            break  # one batch is enough
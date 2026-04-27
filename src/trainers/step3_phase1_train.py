"""
Phase 1 Trainer — UniMol multi-conf training with random duplication.

Design summary:
    - Custom training loop (not MolTrain framework)
    - DataLoader with collate_fn that random-samples K conformers per molecule
    - Pool AFTER classification head (TTA-style) — implemented in MultiConformerUniMol
    - Loss/metric per task type:
        regression: MSE / RMSE
        classification (binary): CrossEntropy / AUC
        multilabel_classification: BCEWithLogits + NaN mask / mean AUC across tasks
    - Best-tracking by valid metric (lower MSE / higher AUC)
    - B2 cache strategy: after restoring best model, do single forward pass
      over all splits with seed=0 fixed sampling → save (N_mol, K, 512) embeddings.

Outputs:
    - phase1_model.pth        — UniMolModel.state_dict() of best epoch
    - cache file (cf. embeddings_cache format)
    - phase1_results.json     — train/valid/test metrics per epoch
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.multi_conf_dataset import (
    IndexedMolKConfDataset,
    MolKConfDataset,
    make_collate_K_indexed_fixed,
    make_collate_K_random_dup,
)
from ..models.multi_conf_unimol import MultiConformerUniMol

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Loss & Metric per task type
# ──────────────────────────────────────────────────────────────────────

class TaskHandler:
    """
    Encapsulates loss, metric, and 'is improvement' logic per task type.

    Convention:
        - higher_better: True for AUC (classification), False for MSE (regression)
        - best metric direction normalized to "higher_better=True" by negating MSE
          internally if needed, so trainer logic is uniform.
    """

    def __init__(self, task_type: str, num_tasks: int = 1):
        self.task_type = task_type
        self.num_tasks = num_tasks

        if task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.metric_name = "rmse"
            self.higher_better = False  # lower MSE/RMSE is better
        elif task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric_name = "auc"
            self.higher_better = True
        elif task_type == "multilabel_classification":
            # Use BCEWithLogitsLoss with NaN mask
            self.loss_fn = self._multilabel_bce_loss
            self.metric_name = "mean_auc"
            self.higher_better = True
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def _multilabel_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        BCEWithLogits with NaN-mask for multilabel.

        logits:  (B, num_tasks)
        targets: (B, num_tasks) — may contain NaN for missing labels
        """
        mask = ~torch.isnan(targets)  # (B, num_tasks) bool
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        targets_filled = torch.where(mask, targets, torch.zeros_like(targets))
        loss_per_elem = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_filled, reduction="none"
        )
        loss = (loss_per_elem * mask.float()).sum() / mask.sum().clamp(min=1.0)
        return loss

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits shape:
            regression               (B, 1)        targets (B,) or (B, 1)
            classification (binary)  (B, 2)        targets (B,) ∈ {0,1} long
            multilabel_classif       (B, n_tasks)  targets (B, n_tasks) float (NaN allowed)
        """
        if self.task_type == "regression":
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return self.loss_fn(logits, targets.float())
        elif self.task_type == "classification":
            return self.loss_fn(logits, targets.long())
        elif self.task_type == "multilabel_classification":
            return self.loss_fn(logits, targets.float())
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def compute_metric(self, logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        logits, targets: numpy arrays, gathered over full split.

        Returns:
            {metric_name: value, ...}
        """
        if self.task_type == "regression":
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            mse = float(np.mean((logits - targets) ** 2))
            rmse = float(np.sqrt(mse))
            return {"mse": mse, "rmse": rmse}
        elif self.task_type == "classification":
            # logits (N, 2), get probs of class 1
            probs = _softmax_np(logits)[:, 1]
            try:
                auc = float(roc_auc_score(targets.astype(np.int32), probs))
            except ValueError:
                # Single-class scenario, AUC undefined
                auc = float("nan")
            return {"auc": auc}
        elif self.task_type == "multilabel_classification":
            # logits (N, n_tasks), targets (N, n_tasks) with NaN
            probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            aucs = []
            per_task = {}
            for t in range(self.num_tasks):
                mask = ~np.isnan(targets[:, t])
                if mask.sum() < 2:
                    per_task[f"auc_task_{t}"] = float("nan")
                    continue
                tgt = targets[mask, t].astype(np.int32)
                if len(np.unique(tgt)) < 2:
                    per_task[f"auc_task_{t}"] = float("nan")
                    continue
                try:
                    a = float(roc_auc_score(tgt, probs[mask, t]))
                    aucs.append(a)
                    per_task[f"auc_task_{t}"] = a
                except ValueError:
                    per_task[f"auc_task_{t}"] = float("nan")
            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            return {"mean_auc": mean_auc, **per_task}
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def is_improvement(self, current: float, best: float) -> bool:
        """Strict improvement check (no tolerance)."""
        if best is None:
            return True
        if self.higher_better:
            return current > best
        else:
            return current < best

    def initial_best(self) -> float:
        """Initial value of 'best so far' before any evaluation."""
        return float("-inf") if self.higher_better else float("inf")


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Phase1Config:
    K: int
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    patience: int = 10
    warmup_ratio: float = 0.03
    max_norm: float = 5.0
    use_amp: bool = True
    weight_decay: float = 0.0
    target_normalize: str = "auto"  # 'auto' | 'none' for regression
    training_seed: int = 42
    fixed_seed_eval: int = 0  # for valid/test conformer sampling


class Phase1Trainer:
    """
    Phase 1: train MultiConformerUniMol with K conformers per molecule.

    After training, restores best model and does B2 cache extraction:
    forward all splits (train/valid/test) with fixed seed=0 sampling,
    returns (N_mol, K, 512) embeddings tensor for each split.
    """

    def __init__(
        self,
        cfg: Phase1Config,
        task_type: str,
        num_tasks: int = 1,
        device: torch.device = torch.device("cuda"),
    ):
        self.cfg = cfg
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.device = device
        self.handler = TaskHandler(task_type, num_tasks)

        # Build model
        logger.info(f"Building MultiConformerUniMol(task={task_type}, num_tasks={num_tasks})")
        self.model = MultiConformerUniMol(
            task_type=task_type,
            num_tasks=num_tasks,
            remove_hs=True,
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model -- {n_params:,} params ({n_trainable:,} trainable)")

    # ─────────────────────────────────────────────────────────────
    # Build DataLoaders
    # ─────────────────────────────────────────────────────────────

    def _build_loaders(
        self,
        train_ds: MolKConfDataset,
        valid_ds: IndexedMolKConfDataset,
        test_ds: IndexedMolKConfDataset,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Generator]:
        """
        Build 3 DataLoaders. Train uses random-dup collate, valid/test deterministic.

        Returns: (train_loader, valid_loader, test_loader, shared_generator)
        """
        cfg = self.cfg

        # Single shared generator (per user choice)
        # Used for both DataLoader.shuffle and collate's random sampling
        torch.manual_seed(cfg.training_seed)
        np.random.seed(cfg.training_seed)
        shared_gen = torch.Generator()
        shared_gen.manual_seed(cfg.training_seed)

        # Train: random dup, shuffle on
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            generator=shared_gen,
            collate_fn=make_collate_K_random_dup(cfg.K, shared_gen),
        )

        # Valid/test: deterministic per-molecule sampling, no shuffle
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

        return train_loader, valid_loader, test_loader, shared_gen

    # ─────────────────────────────────────────────────────────────
    # Optimizer + scheduler
    # ─────────────────────────────────────────────────────────────

    def _build_optim_and_scheduler(self, n_train_steps: int):
        cfg = self.cfg
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        # Linear warmup + linear decay (mimic MolTrain default)
        warmup_steps = int(n_train_steps * cfg.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(0.0, (n_train_steps - step) / max(1, n_train_steps - warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    # ─────────────────────────────────────────────────────────────
    # Target normalization (regression only)
    # ─────────────────────────────────────────────────────────────

    def _setup_target_scaler(self, train_targets: np.ndarray):
        """
        For regression: standardize targets using train mean/std.
        For classification: no-op.

        Sets self.target_mean, self.target_std used in {scale, unscale} targets.
        """
        cfg = self.cfg
        if self.task_type != "regression" or cfg.target_normalize == "none":
            self.target_mean = 0.0
            self.target_std = 1.0
            return
        self.target_mean = float(np.mean(train_targets))
        self.target_std = float(np.std(train_targets))
        if self.target_std < 1e-6:
            logger.warning(
                f"target_std={self.target_std:.6f} too small, setting to 1.0"
            )
            self.target_std = 1.0
        logger.info(
            f"Target normalize -- mean={self.target_mean:.4f}, std={self.target_std:.4f}"
        )

    def _scale_targets(self, y: torch.Tensor) -> torch.Tensor:
        if self.task_type != "regression" or self.target_std == 1.0:
            return y
        return (y - self.target_mean) / self.target_std

    def _unscale_predictions(self, pred: np.ndarray) -> np.ndarray:
        if self.task_type != "regression" or self.target_std == 1.0:
            return pred
        return pred * self.target_std + self.target_mean

    # ─────────────────────────────────────────────────────────────
    # Train one epoch
    # ─────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer,
        scheduler,
        scaler,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            targets = self._scale_targets(batch["targets"])

            optimizer.zero_grad(set_to_none=True)

            if self.cfg.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(
                        batch["src_tokens"],
                        batch["src_distance"],
                        batch["src_coord"],
                        batch["src_edge_type"],
                    )
                    loss = self.handler.compute_loss(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = self.model(
                    batch["src_tokens"],
                    batch["src_distance"],
                    batch["src_coord"],
                    batch["src_edge_type"],
                )
                loss = self.handler.compute_loss(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ─────────────────────────────────────────────────────────────
    # Evaluate (compute predictions over full split)
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """
        Forward all batches, gather predictions and targets, compute metric.

        Returns:
            {
                'preds': np.ndarray (N_mol, output_dim) — UNSCALED
                'targets': np.ndarray (N_mol,) or (N_mol, n_tasks)
                'metrics': dict from TaskHandler.compute_metric
            }
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in loader:
            batch = self._batch_to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                logits = self.model(
                    batch["src_tokens"],
                    batch["src_distance"],
                    batch["src_coord"],
                    batch["src_edge_type"],
                )  # (B, output_dim) or (B, output_dim) for regression with squeeze
            all_preds.append(logits.detach().float().cpu().numpy())
            all_targets.append(batch["targets"].detach().cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Unscale for regression
        if self.task_type == "regression":
            if preds.ndim == 2 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)
            preds = self._unscale_predictions(preds)

        metrics = self.handler.compute_metric(preds, targets)
        return {"preds": preds, "targets": targets, "metrics": metrics}

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    # ─────────────────────────────────────────────────────────────
    # B2 cache extraction (after restoring best model)
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_cls_cache(
        self,
        dataset: IndexedMolKConfDataset,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward all molecules through best model, extract CLS embeddings.

        Uses the SAME deterministic seed=0 sampling that was used during valid/test
        eval, ensuring cache and final eval are consistent (B2 strategy).

        Returns:
            {
                'X':            (N_mol, K, 512) float32 cpu
                'valid_counts': (N_mol,)
                'y':            (N_mol,) or (N_mol, n_tasks)
                'kept_indices': (N_mol,) — within-dataset indices (== range(N_mol) after filter)
            }
        """
        self.model.eval()
        cfg = self.cfg

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_K_indexed_fixed(cfg.K, fixed_seed=cfg.fixed_seed_eval),
        )

        N_mol = len(dataset)
        D = 512  # UniMol encoder embed dim
        X_full = torch.zeros(N_mol, cfg.K, D, dtype=torch.float32)

        for batch in loader:
            batch_dev = self._batch_to_device(batch)
            cls = self.model.extract_cls(
                batch_dev["src_tokens"],
                batch_dev["src_distance"],
                batch_dev["src_coord"],
                batch_dev["src_edge_type"],
            )  # (B, K, 512)
            mol_indices = batch["mol_indices"].cpu()
            X_full[mol_indices] = cls.detach().cpu().float()

        return {
            "X":             X_full,
            "valid_counts":  torch.from_numpy(dataset.valid_counts),
            "y":             torch.from_numpy(np.asarray(dataset.targets, dtype=np.float32)),
            "kept_indices":  torch.from_numpy(dataset.kept_indices),
        }

    # ─────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        train_ds: MolKConfDataset,
        valid_ds: IndexedMolKConfDataset,
        test_ds: IndexedMolKConfDataset,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Run phase 1: train + restore best + extract embeddings cache.

        Returns dict with keys:
            'best_epoch', 'history', 'model_path', 'metrics',
            'cache_train', 'cache_valid', 'cache_test'
        """
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)

        # 1. Setup target scaler (for regression)
        if self.task_type == "regression":
            self._setup_target_scaler(np.asarray(train_ds.targets))
        else:
            self.target_mean = 0.0
            self.target_std = 1.0

        # 2. Build loaders
        train_loader, valid_loader, test_loader, shared_gen = self._build_loaders(
            train_ds, valid_ds, test_ds
        )
        n_train_steps = len(train_loader) * cfg.epochs

        # 3. Optimizer + scheduler + AMP scaler
        optimizer, scheduler = self._build_optim_and_scheduler(n_train_steps)
        amp_scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

        # 4. Train loop
        best_metric = self.handler.initial_best()
        best_state = None
        best_epoch = -1
        patience_ctr = 0
        history = []
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"Phase 1: UniMol Multi-Conf Training")
        print(f"Task: {self.task_type} | K: {cfg.K} | Device: {self.device}")
        print(f"Epochs: {cfg.epochs} | Patience: {cfg.patience} | Batch: {cfg.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(cfg.epochs):
            t_ep = time.time()
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, amp_scaler)
            valid_eval = self._evaluate(valid_loader)
            valid_metric = valid_eval["metrics"][self.handler.metric_name]

            improved = self.handler.is_improvement(valid_metric, best_metric)
            marker = "★" if improved else " "
            ep_time = time.time() - t_ep

            log_line = (
                f"{marker} Epoch {epoch:3d}  "
                f"train_loss={train_loss:.4f}  "
                f"valid_{self.handler.metric_name}={valid_metric:.4f}  "
                f"[{ep_time:.1f}s]"
            )
            if improved:
                best_metric = valid_metric
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                log_line += f"  (patience {patience_ctr}/{cfg.patience})"

            print(log_line)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                f"valid_{self.handler.metric_name}": valid_metric,
                "improved": improved,
                "elapsed_s": ep_time,
                **{f"valid_{k}": v for k, v in valid_eval["metrics"].items()},
            })

            if patience_ctr >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

        total_time = time.time() - t0
        print(f"\nTotal training time: {total_time:.1f}s")

        # 5. Restore best
        if best_state is None:
            raise RuntimeError("No best state recorded — training failed")
        self.model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} "
              f"(valid {self.handler.metric_name}={best_metric:.4f})")

        # 6. Final evaluation on all splits with best model
        # Use FIXED sampling for all 3 splits (deterministic, matches cache)
        print("\nEvaluating with best model (fixed seed=0 sampling)...")

        # Build deterministic train loader for final eval (matches valid/test pattern)
        train_ds_indexed = self._wrap_indexed(train_ds)
        train_loader_fixed = DataLoader(
            train_ds_indexed,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_K_indexed_fixed(cfg.K, fixed_seed=cfg.fixed_seed_eval),
        )

        train_eval = self._evaluate(train_loader_fixed)
        valid_eval = self._evaluate(valid_loader)
        test_eval = self._evaluate(test_loader)

        print(f"  Train {self.handler.metric_name}: "
              f"{train_eval['metrics'][self.handler.metric_name]:.4f}")
        print(f"  Valid {self.handler.metric_name}: "
              f"{valid_eval['metrics'][self.handler.metric_name]:.4f}")
        print(f"  Test  {self.handler.metric_name}: "
              f"{test_eval['metrics'][self.handler.metric_name]:.4f}")

        # 7. Save model
        model_path = os.path.join(output_dir, "phase1_model.pth")
        torch.save(self.model.state_dict_base_only(), model_path)
        print(f"\nModel saved → {model_path}")

        # 8. B2 strategy: extract embeddings cache for all splits
        print("\nExtracting embeddings cache (B2 strategy)...")
        cache_train = self._extract_cls_cache(self._wrap_indexed(train_ds))
        cache_valid = self._extract_cls_cache(valid_ds)
        cache_test = self._extract_cls_cache(test_ds)
        print(f"  Cache shapes: "
              f"train {tuple(cache_train['X'].shape)}, "
              f"valid {tuple(cache_valid['X'].shape)}, "
              f"test {tuple(cache_test['X'].shape)}")

        return {
            "best_epoch": best_epoch,
            "best_valid_metric": best_metric,
            "history": history,
            "model_path": model_path,
            "metrics": {
                "train": train_eval["metrics"],
                "valid": valid_eval["metrics"],
                "test":  test_eval["metrics"],
            },
            "cache_train": cache_train,
            "cache_valid": cache_valid,
            "cache_test":  cache_test,
            "total_time_s": total_time,
            "target_mean": self.target_mean,
            "target_std":  self.target_std,
        }

    @staticmethod
    def _wrap_indexed(ds: MolKConfDataset) -> IndexedMolKConfDataset:
        """Wrap a plain MolKConfDataset as IndexedMolKConfDataset (for cache extraction)."""
        if isinstance(ds, IndexedMolKConfDataset):
            return ds
        wrapped = IndexedMolKConfDataset.__new__(IndexedMolKConfDataset)
        wrapped.conformers_per_mol = ds.conformers_per_mol
        wrapped.targets = ds.targets
        wrapped.kept_indices = ds.kept_indices
        wrapped.valid_counts = ds.valid_counts
        return wrapped
"""
MultiTreeGPModel — Plan A encoding cho Step 3.

Kiến trúc:
    K Forests, mỗi Forest:
        - pop_size = N
        - input_len = D (512)
        - output_len = num_trees_per_conformer (multi-tree: num_trees_per_conformer subtrees độc lập per individual)
        - max_tree_len = configured (e.g. 640 cho num_trees_per_conformer=10, max_layer_cnt=5)

    Super-individual i = (F_1[i], F_2[i], ..., F_K[i])  — index-aligned coevolution.

Forward pass:
    X: (N_mol, K, D)
    For k = 1..K:
        Z_k = F_k.batch_forward(X[:, k, :])   # (pop_size, N_mol, num_trees_per_conformer)
    Stack → Z (pop_size, K, N_mol, num_trees_per_conformer)

Sequential ridge:
    1. Per-conformer inner: fit (Z_k, y) batched → w_inner^(k), b_inner^(k)
       s_k = Z_k · w_inner^(k) + b_inner^(k)
    2. Stack S (pop, N_mol, K)
    3. Outer: fit (S, y) batched → w_outer, b_outer
       y_pred = S · w_outer + b_outer

Fitness:
    fitness = -MSE (higher=better). NaN → -1e10.

API public chính:
    compute_features(X)                              — forward GP
    evaluate_fitness(X, y)                           — fit+eval trên cùng data
    evaluate_cross(X_fit, y_fit, X_eval, y_eval)    — fit trên 1, eval trên 2
    evaluate_dual(X_fit, y_fit, X_eval, y_eval)     — efficient: fit ONCE, eval BOTH
    predict_with_ridges(X_eval, ridges)             — apply pre-fit ridges trên X mới
    evolve_step(fitness)                             — GP step tất cả K Forests
    get_individual_ridges(ridges_batched, idx)      — slice ridge params of 1 individual
    snapshot_individual_trees(idx)                   — save trees' node tensors
    get_sympy_formulas(idx)                          — K lists of num_trees_per_conformer sympy expressions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from evogp.algorithm import (
    DefaultCrossover,
    DefaultMutation,
    DefaultSelection,
    GeneticProgramming,
)
from evogp.tree import Forest, GenerateDescriptor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GPConfig:
    K: int
    num_trees_per_conformer: int
    D: int = 512
    pop_size: int = 500
    max_tree_len: int = 640
    max_layer_cnt: int = 5
    mutation_max_layer_cnt: int = 3
    using_funcs: Tuple[str, ...] = ("+", "-", "*", "/")
    const_samples: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    mutation_rate: float = 0.2
    survival_rate: float = 0.3
    elite_rate: float = 0.01
    parsimony_alpha: float = 0.0
    lambda_inner: float = 1.0
    lambda_outer: float = 0.1
    nan_fitness: float = -1e10


# ──────────────────────────────────────────────────────────────────────
# Batched ridge closed-form
# ──────────────────────────────────────────────────────────────────────

def _batched_ridge_solve(
    X: torch.Tensor,       # (B, N, D)
    y: torch.Tensor,       # (B, N) hoặc (N,) broadcast
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched ridge closed-form với bias term augmented:
        minimize ||X w + b - y||^2 + λ(||w||^2 + b^2)

    Bias regularize cùng w (simplification). Với λ điển hình (0.1-1.0) và
    bias scale nhỏ, ảnh hưởng không đáng kể.

    Returns:
        w: (B, D)
        b: (B,)
    """
    B, N, D = X.shape
    if y.ndim == 1:
        y = y.unsqueeze(0).expand(B, -1)
    assert y.shape == (B, N), f"y shape {y.shape} incompatible with X {X.shape}"

    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ones = torch.ones(B, N, 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([X, ones], dim=-1)                                 # (B, N, D+1)

    XtX = torch.bmm(X_aug.transpose(1, 2), X_aug)                        # (B, D+1, D+1)
    Xty = torch.bmm(X_aug.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)  # (B, D+1)

    eye = torch.eye(D + 1, device=X.device, dtype=X.dtype).unsqueeze(0)
    A = XtX + lam * eye

    try:
        w_full = torch.linalg.solve(A, Xty.unsqueeze(-1)).squeeze(-1)    # (B, D+1)
    except Exception:
        w_full = torch.zeros_like(Xty)
        for i in range(B):
            try:
                w_full[i] = torch.linalg.lstsq(A[i], Xty[i]).solution
            except Exception:
                w_full[i] = 0.0

    return w_full[:, :D], w_full[:, D]


# ──────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────

class MultiTreeGPModel:
    """K Forests song song, index-aligned coevolution."""

    def __init__(self, config: GPConfig, device: torch.device = torch.device("cuda")):
        self.cfg = config
        self.device = device

        self.descriptor = GenerateDescriptor(
            max_tree_len=config.max_tree_len,
            input_len=config.D,
            output_len=config.num_trees_per_conformer,
            using_funcs=list(config.using_funcs),
            max_layer_cnt=config.max_layer_cnt,
            const_samples=list(config.const_samples),
        )
        self.mutation_descriptor = self.descriptor.update(
            max_layer_cnt=config.mutation_max_layer_cnt
        )

        self.algorithms: List[GeneticProgramming] = []
        for _ in range(config.K):
            forest = Forest.random_generate(
                pop_size=config.pop_size, descriptor=self.descriptor
            )
            algo = GeneticProgramming(
                initial_forest=forest,
                crossover=DefaultCrossover(),
                mutation=DefaultMutation(
                    mutation_rate=config.mutation_rate,
                    descriptor=self.mutation_descriptor,
                ),
                selection=DefaultSelection(
                    survival_rate=config.survival_rate,
                    elite_rate=config.elite_rate,
                ),
            )
            self.algorithms.append(algo)

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def compute_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N_mol, K, D)  →  Z: (pop_size, K, N_mol, num_trees_per_conformer)

        Dùng CURRENT forests (sau evolve_step).
        """
        assert X.shape[1] == self.cfg.K
        assert X.shape[2] == self.cfg.D
        X = X.to(self.device)

        Z_list = []
        for k in range(self.cfg.K):
            x_k = X[:, k, :].contiguous()
            z_k = self.algorithms[k].forest.batch_forward(x_k)    # (pop, N_mol, num_trees_per_conformer)
            Z_list.append(z_k)
        return torch.stack(Z_list, dim=1)                          # (pop, K, N_mol, num_trees_per_conformer)

    # ─────────────────────────────────────────────────────────────
    # Internal batched helpers
    # ─────────────────────────────────────────────────────────────

    def _fit_ridges(self, Z: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fit sequential ridges.

        Z:  (pop, K, N, num_trees_per_conformer)
        y:  (N,)

        Returns:
            {
                'w_in':  (pop*K, num_trees_per_conformer),    # flat_idx = p*K + k
                'b_in':  (pop*K,),
                'w_out': (pop, K),
                'b_out': (pop,),
            }
        """
        cfg = self.cfg
        pop, K, N, num_trees = Z.shape
        assert K == cfg.K and num_trees == cfg.num_trees_per_conformer

        # Inner ridge (per-conformer) batched
        Z_flat = Z.reshape(pop * K, N, num_trees)
        w_in, b_in = _batched_ridge_solve(Z_flat, y, lam=cfg.lambda_inner)

        s_flat = torch.einsum("bnq,bq->bn", Z_flat, w_in) + b_in.unsqueeze(1)
        S = s_flat.reshape(pop, K, N).transpose(1, 2)              # (pop, N, K)

        # Outer ridge batched
        w_out, b_out = _batched_ridge_solve(S, y, lam=cfg.lambda_outer)

        return {"w_in": w_in, "b_in": b_in, "w_out": w_out, "b_out": b_out}

    def _predict(self, Z: torch.Tensor, ridges: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply pre-fit ridges lên features Z.

        Z:      (pop, K, N, num_trees_per_conformer)
        ridges: dict từ _fit_ridges (phải match pop)

        Returns: y_pred (pop, N)
        """
        pop, K, N, num_trees = Z.shape
        w_in, b_in = ridges["w_in"], ridges["b_in"]
        w_out, b_out = ridges["w_out"], ridges["b_out"]

        Z_flat = Z.reshape(pop * K, N, num_trees)
        Z_flat = torch.nan_to_num(Z_flat, nan=0.0, posinf=0.0, neginf=0.0)

        s_flat = torch.einsum("bnq,bq->bn", Z_flat, w_in) + b_in.unsqueeze(1)
        S = s_flat.reshape(pop, K, N).transpose(1, 2)              # (pop, N, K)

        y_pred = torch.einsum("bnk,bk->bn", S, w_out) + b_out.unsqueeze(1)
        return y_pred

    def _mse_fitness(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """y_pred: (pop, N), y_target: (N,). Returns fitness = -MSE, NaN-safe."""
        cfg = self.cfg
        err = y_pred - y_target.to(y_pred.device).unsqueeze(0)
        err = torch.nan_to_num(err, nan=1e6, posinf=1e6, neginf=1e6)
        mse = (err ** 2).mean(dim=1)
        fitness = -mse

        if cfg.parsimony_alpha > 0.0:
            sizes = self._avg_tree_size_per_individual()
            fitness = fitness - cfg.parsimony_alpha * sizes

        return torch.nan_to_num(fitness, nan=cfg.nan_fitness,
                                posinf=cfg.nan_fitness, neginf=cfg.nan_fitness)

    # ─────────────────────────────────────────────────────────────
    # Public evaluation APIs
    # ─────────────────────────────────────────────────────────────

    def evaluate_fitness(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        return_ridges: bool = False,
    ):
        """Fit ridges trên (X, y) và evaluate trên cùng data. Returns fitness (pop,)."""
        y = y.to(self.device)
        Z = self.compute_features(X)
        ridges = self._fit_ridges(Z, y)
        y_pred = self._predict(Z, ridges)
        fitness = self._mse_fitness(y_pred, y)
        if return_ridges:
            return fitness, ridges
        return fitness

    def evaluate_cross(
        self,
        X_fit: torch.Tensor,
        y_fit: torch.Tensor,
        X_eval: torch.Tensor,
        y_eval: torch.Tensor,
        return_ridges: bool = False,
    ):
        """Fit trên (X_fit, y_fit), evaluate trên (X_eval, y_eval)."""
        y_fit = y_fit.to(self.device)
        y_eval = y_eval.to(self.device)

        Z_fit = self.compute_features(X_fit)
        ridges = self._fit_ridges(Z_fit, y_fit)

        Z_eval = self.compute_features(X_eval)
        y_pred = self._predict(Z_eval, ridges)
        fitness = self._mse_fitness(y_pred, y_eval)

        if return_ridges:
            return fitness, ridges
        return fitness

    def evaluate_dual(
        self,
        X_fit: torch.Tensor,
        y_fit: torch.Tensor,
        X_eval: torch.Tensor,
        y_eval: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Efficient: fit ridges ONCE, eval trên CẢ fit data và eval data.

        Returns:
            fit_fitness:  (pop,) — on (X_fit, y_fit), dùng cho evolve
            eval_fitness: (pop,) — on (X_eval, y_eval), dùng cho best-of-run
            ridges:       dict batched — dùng cho test prediction sau này
        """
        y_fit = y_fit.to(self.device)
        y_eval = y_eval.to(self.device)

        Z_fit = self.compute_features(X_fit)
        ridges = self._fit_ridges(Z_fit, y_fit)

        y_pred_fit = self._predict(Z_fit, ridges)
        fit_fitness = self._mse_fitness(y_pred_fit, y_fit)

        Z_eval = self.compute_features(X_eval)
        y_pred_eval = self._predict(Z_eval, ridges)
        eval_fitness = self._mse_fitness(y_pred_eval, y_eval)

        return fit_fitness, eval_fitness, ridges

    def predict_with_ridges(
        self,
        X_eval: torch.Tensor,
        ridges: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply pre-fit ridges lên X_eval. Returns (pop, N_eval)."""
        Z_eval = self.compute_features(X_eval)
        return self._predict(Z_eval, ridges)

    # ─────────────────────────────────────────────────────────────
    # Evolution step
    # ─────────────────────────────────────────────────────────────

    def evolve_step(self, fitness: torch.Tensor) -> None:
        """Step tất cả K Forests với cùng fitness (index-aligned)."""
        fitness = fitness.to(self.device)
        for k in range(self.cfg.K):
            self.algorithms[k].step(fitness)

    # ─────────────────────────────────────────────────────────────
    # Per-individual extraction (best-of-run)
    # ─────────────────────────────────────────────────────────────

    def get_individual_ridges(
        self,
        ridges_batched: Dict[str, torch.Tensor],
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Slice ridge params cho super-individual idx từ batched tensors.

        Layout: w_in flat idx = p*K + k → individual `idx` có inner ridges tại
        [idx*K, idx*K+K-1]. Outer is direct index.
        """
        K = self.cfg.K
        return {
            "w_inner": ridges_batched["w_in"][idx * K:(idx + 1) * K].detach().cpu().clone(),
            "b_inner": ridges_batched["b_in"][idx * K:(idx + 1) * K].detach().cpu().clone(),
            "w_outer": ridges_batched["w_out"][idx].detach().cpu().clone(),
            "b_outer": ridges_batched["b_out"][idx].detach().cpu().clone(),
        }

    def get_sympy_formulas(self, idx: int) -> List:
        """Return list K phần tử; mỗi element là list num_trees_per_conformer sympy expressions hoặc error string."""
        out = []
        for k in range(self.cfg.K):
            tree = self.algorithms[k].forest[idx]
            try:
                exprs = tree.to_sympy_expr()
            except Exception as e:
                exprs = f"<error: {type(e).__name__}: {e}>"
            out.append(exprs)
        return out

    def snapshot_individual_trees(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        """Snapshot node tensors của K trees (individual idx) cho save/load."""
        state = []
        for k in range(self.cfg.K):
            tree = self.algorithms[k].forest[idx]
            state.append({
                "node_type":    tree.node_type.detach().cpu().clone(),
                "node_value":   tree.node_value.detach().cpu().clone(),
                "subtree_size": tree.subtree_size.detach().cpu().clone(),
            })
        return state

    # ─────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────

    def _avg_tree_size_per_individual(self) -> torch.Tensor:
        """Average non-null node count across K forests — (pop,) on device."""
        sizes = []
        for k in range(self.cfg.K):
            nt = self.algorithms[k].forest.batch_node_type
            sizes.append((nt != 0).sum(dim=1).float())
        return torch.stack(sizes, dim=0).mean(dim=0)
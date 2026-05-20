"""
EGGROLL: Evolution Guided GeneRal Optimisation via Low-rank Learning
PyTorch implementation following Sarkar et al. (2026), Algorithm 1.

This module provides:
  - EGGROLLOptimizer: samples perturbations, computes gradient estimate via
    score-weighted average, and applies via any torch.optim.Optimizer.

For Experiment A (frozen encoder + small head), we use FULL perturbation
(noise tensor has same shape as parameter; no LoRA needed since head is small).

For Experiment B (encoder fine-tuning), this will be paired with a LoRALinear
wrapper that applies low-rank perturbation E = sigma/sqrt(r) * A @ B^T to
encoder Linear layers. The EGGROLLOptimizer itself stays the same — it just
optimizes the (A, B) parameters instead of full weight matrices.

Reference: paper Algorithm 1
    For each worker i (in parallel):
        sample E_i ~ p(E)
        f_i = f(W = M + sigma * E_i)
    Update:
        M = M + (alpha / N) * sum_i E_i * f_i
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn


class EGGROLLOptimizer:
    """
    EGGROLL gradient estimator + perturbation sampler.

    Typical training-step usage:
        noises    = optimizer.sample_population()
        # User: evaluate population externally, producing tensor of fitnesses
        # with shape (pop_size,). Higher = better (maximization objective).
        fitnesses = evaluate_population(noises)
        optimizer.step(noises, fitnesses, torch_optimizer)

    The torch_optimizer (e.g. Adam) applies EGGROLL-estimated gradients so we
    inherit momentum / adaptive learning rates from standard PyTorch optimizers.

    Args:
        params: list of nn.Parameter to optimize via EGGROLL.
        sigma: noise scale. Typical 0.001 to 0.1 depending on task and param norm.
        population_size: N. Must be even if use_antithetic=True.
        use_antithetic: if True, samples pop_size // 2 directions then mirrors
            (+noise, -noise pairs). Reduces variance of gradient estimate.
        fitness_shaping: 'zscore' (recommended), 'rank', or 'raw'.

    Note on signs:
        Fitness is MAXIMIZED. We compute grad_J = ∇_M E[f(M + sigma * E)],
        then NEGATE it before handing to torch optimizers (which MINIMIZE).
        So you set fitness = -loss (or fitness = R², Pearson, etc.) and let
        the torch optimizer do its thing.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        sigma: float,
        population_size: int,
        use_antithetic: bool = True,
        fitness_shaping: Literal["zscore", "rank", "raw"] = "zscore",
    ):
        if use_antithetic and population_size % 2 != 0:
            raise ValueError("population_size must be even when use_antithetic=True")
        self.params = list(params)
        self.sigma = float(sigma)
        self.pop_size = int(population_size)
        self.use_antithetic = bool(use_antithetic)
        self.fitness_shaping = fitness_shaping

    # ------------------------------------------------------------------
    # Population sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_population(
        self, generator: Optional[torch.Generator] = None
    ) -> List[torch.Tensor]:
        """Sample noise tensors. Returns list (one per param) of shape (pop_size, *p.shape)."""
        n_unique = self.pop_size // 2 if self.use_antithetic else self.pop_size
        noises: List[torch.Tensor] = []
        for p in self.params:
            base = torch.randn(
                (n_unique,) + tuple(p.shape),
                device=p.device,
                dtype=p.dtype,
                generator=generator,
            )
            if self.use_antithetic:
                noise = torch.cat([base, -base], dim=0)
            else:
                noise = base
            noises.append(noise)
        return noises

    @torch.no_grad()
    def get_perturbed_params(
        self, noises: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Returns perturbed params batched over population.

        Shape: list of tensors, each (pop_size, *p.shape).
        Useful when you can vectorize the population forward pass (e.g. the
        whole model is one linear layer; see Experiment A).
        """
        return [
            p.detach().unsqueeze(0) + self.sigma * noise
            for p, noise in zip(self.params, noises)
        ]

    # ------------------------------------------------------------------
    # Fitness shaping
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _shape_fitness(self, fitnesses: torch.Tensor) -> torch.Tensor:
        if self.fitness_shaping == "raw":
            return fitnesses
        if self.fitness_shaping == "zscore":
            return (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        if self.fitness_shaping == "rank":
            # rank-based mapping to [-0.5, 0.5]
            ranks = torch.argsort(torch.argsort(fitnesses)).to(fitnesses.dtype)
            return ranks / (len(fitnesses) - 1) - 0.5
        raise ValueError(f"Unknown fitness_shaping: {self.fitness_shaping}")

    # ------------------------------------------------------------------
    # Gradient estimation & update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_gradients(
        self, noises: List[torch.Tensor], fitnesses: torch.Tensor
    ) -> List[torch.Tensor]:
        """Estimate gradients via score-weighted average.

        Returns: list of tensors same shape as each param.
        NEGATED for compatibility with minimizing torch optimizers.
        """
        f = self._shape_fitness(fitnesses.detach().to(noises[0].device))
        grads: List[torch.Tensor] = []
        for noise in noises:
            # ∇J ≈ (1 / (N * sigma)) * Σ_i f_i * noise_i
            grad_J = torch.einsum("p,p...->...", f, noise) / (self.pop_size * self.sigma)
            grads.append(-grad_J)  # negate for minimization
        return grads

    @torch.no_grad()
    def step(
        self,
        noises: List[torch.Tensor],
        fitnesses: torch.Tensor,
        torch_optimizer: torch.optim.Optimizer,
    ) -> List[torch.Tensor]:
        """Compute gradients and apply via torch_optimizer.step().

        Returns the gradients (mainly for logging / debugging).
        """
        grads = self.compute_gradients(noises, fitnesses)
        for p, g in zip(self.params, grads):
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.copy_(g)
        torch_optimizer.step()
        return grads


# ----------------------------------------------------------------------
# Helper for Experiment A: vectorized population eval on a Linear head
# ----------------------------------------------------------------------
@torch.no_grad()
def linear_head_population_predict(
    X: torch.Tensor,
    weight_pop: torch.Tensor,
    bias_pop: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a population of Linear(in_dim -> 1) heads in parallel.

    Args:
        X:          (N, in_dim) — features (e.g. cached Unimol cls_repr).
        weight_pop: (pop_size, 1, in_dim) — perturbed weights for population.
        bias_pop:   (pop_size, 1) — perturbed biases for population.

    Returns:
        preds: (pop_size, N) — predictions per candidate per sample.
    """
    # X @ W^T: (N, in_dim) @ (pop_size, in_dim, 1) -> (pop_size, N, 1)
    # Using einsum for clarity:
    preds = torch.einsum("nd,pod->pn", X, weight_pop) + bias_pop  # bias broadcasts
    return preds
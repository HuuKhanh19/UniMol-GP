"""
EGGROLL for encoder fine-tuning with LoRA-style low-rank perturbation.

Follows EGGROLL paper Algorithm 1:
    E_i = (sigma / sqrt(r)) * A_i @ B_i^T  with A_i, B_i ~ N(0, I) i.i.d.
    W_i = M + sigma * E_i = M + (sigma^2 / sqrt(r)) * A_i @ B_i^T

WAIT — re-reading the paper Eq (4): E_i = (1/sqrt(r)) * A_i @ B_i^T
The candidate is evaluated at W_i = M + sigma * E_i.
So perturbation magnitude is sigma * (1/sqrt(r)) * A @ B^T.

Update (paper Eq 3): M = M + (alpha / N) * sum_i E_i * f_i
                       = M + (alpha / (N * sqrt(r))) * sum_i A_i @ B_i^T * f_i

This module provides:
  - EGGROLLEncoderState: manages M (learnable full weights) + sampling + gradient
  - make_vmap_forward: returns a vmap'd forward function over LoRA candidates
  - chunked_vmap_call: chunks the population to fit in GPU memory

The Adam-style outer optimizer is applied AFTER computing EGGROLL gradient,
giving us momentum & adaptive lr "for free".
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.func as tfunc


# ----------------------------------------------------------------------
# State container
# ----------------------------------------------------------------------
class EGGROLLEncoderState(nn.Module):
    """Manages learnable M[name] (full weight matrices) + LoRA noise sampling.

    M[name] starts at pre-trained values and is updated via EGGROLL gradient
    estimate.  A and B are NOISE (sampled fresh per step), not learned.

    Layout:
        target_param_names: list of full dotted paths, e.g.
            ['encoder.layers.0.self_attn.q_proj.weight', ...]
        M: ParameterDict mapping safe_name (no dots) -> nn.Parameter
            with requires_grad=False (we manually set .grad before optimizer.step)
        name_to_safe / safe_to_name: bijection for safe ParameterDict keys.

    Sampled per step:
        A_pop[name]: (pop, out, rank)
        B_pop[name]: (pop, rank, in)
    """

    def __init__(
        self,
        model: nn.Module,
        target_param_names: List[str],
        rank: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.rank = int(rank)
        self.sigma = float(sigma)
        self.device = device
        self.dtype = dtype
        self.target_param_names = list(target_param_names)

        # safe name = original name with dots replaced (ParameterDict can't have dots)
        self.name_to_safe = {n: n.replace(".", "__") for n in self.target_param_names}
        self.safe_to_name = {v: k for k, v in self.name_to_safe.items()}

        # Cache pre-trained weights as the initial M
        state = dict(model.named_parameters())
        self.M = nn.ParameterDict()
        self.shapes: Dict[str, Tuple[int, int]] = {}
        for name in self.target_param_names:
            if name not in state:
                raise KeyError(
                    f"Target param '{name}' not found in model. "
                    f"Available top-level: {sorted(list(state.keys()))[:5]} ..."
                )
            w = state[name].detach().clone().to(device=device, dtype=dtype)
            if w.dim() != 2:
                raise ValueError(
                    f"Target '{name}' must be 2D (Linear.weight), got {w.dim()}D shape {w.shape}"
                )
            # requires_grad=False because we manually set .grad (no autograd backward)
            self.M[self.name_to_safe[name]] = nn.Parameter(w, requires_grad=False)
            self.shapes[name] = (w.shape[0], w.shape[1])

        n_params = sum(s[0] * s[1] for s in self.shapes.values())
        print(f"  EGGROLLEncoderState: {len(self.target_param_names)} targets, "
              f"{n_params/1e6:.2f}M base params, rank={rank}, sigma={sigma}")

    # ------------------------------------------------------------------
    # Noise sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_noise(
        self,
        pop_size: int,
        use_antithetic: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample fresh (A_pop, B_pop) for one EGGROLL step."""
        if use_antithetic and pop_size % 2 != 0:
            raise ValueError(f"pop_size must be even when antithetic=True, got {pop_size}")
        n_unique = pop_size // 2 if use_antithetic else pop_size
        A_pop, B_pop = {}, {}
        for name in self.target_param_names:
            out, in_dim = self.shapes[name]
            a = torch.randn(n_unique, out, self.rank, device=self.device,
                            dtype=self.dtype, generator=generator)
            b = torch.randn(n_unique, self.rank, in_dim, device=self.device,
                            dtype=self.dtype, generator=generator)
            if use_antithetic:
                a = torch.cat([a, -a], dim=0)
                b = torch.cat([b, -b], dim=0)
            A_pop[name] = a
            B_pop[name] = b
        return A_pop, B_pop

    # ------------------------------------------------------------------
    # Gradient computation (no backward needed)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_and_assign_grads(
        self,
        A_pop: Dict[str, torch.Tensor],
        B_pop: Dict[str, torch.Tensor],
        shaped_fitnesses: torch.Tensor,
    ):
        """Compute ∇_M J ≈ (1/(N*sqrt(r))) Σ_i f_i (A_i @ B_i^T), negate, assign to .grad.

        After this, call your outer optimizer.step().
        """
        pop = shaped_fitnesses.shape[0]
        # NOTE: paper uses E_i = (1/sqrt(r)) A @ B^T (no sigma).
        # ∇_M J = (1/sigma) * E[f * E_i] ≈ (1/(N*sigma)) Σ_i f_i * (1/sqrt(r)) A_i B_i^T
        # Adam will then minimize (-J), so we negate.
        for name in self.target_param_names:
            # Σ_i f_i * (A_i @ B_i^T): use einsum to avoid materializing per-candidate matrices
            grad_J = torch.einsum(
                "p,por,pri->oi",
                shaped_fitnesses.to(A_pop[name].device, A_pop[name].dtype),
                A_pop[name], B_pop[name],
            )
            grad_J = grad_J / (pop * self.sigma * (self.rank ** 0.5))
            grad_for_min = -grad_J  # for torch optimizers that minimize
            safe = self.name_to_safe[name]
            self.M[safe].grad = grad_for_min.detach().clone()

    # ------------------------------------------------------------------
    # Build state_dict override for functional_call (one candidate)
    # ------------------------------------------------------------------
    def build_overrides(
        self,
        A_one: Dict[str, torch.Tensor],
        B_one: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Override dict for ONE candidate: W_i = M + (sigma/sqrt(r)) * A @ B."""
        scale = self.sigma / (self.rank ** 0.5)
        overrides = {}
        for name in self.target_param_names:
            safe = self.name_to_safe[name]
            E = A_one[name] @ B_one[name]  # (out, in)
            overrides[name] = self.M[safe] + scale * E
        return overrides

    @torch.no_grad()
    def current_weights(self) -> Dict[str, torch.Tensor]:
        """Get current M (un-perturbed) — useful for evaluation."""
        return {name: self.M[self.name_to_safe[name]].detach().clone()
                for name in self.target_param_names}


# ----------------------------------------------------------------------
# vmap forward factory
# ----------------------------------------------------------------------
def make_vmap_forward(
    model: nn.Module,
    state: EGGROLLEncoderState,
    output_extractor: Callable[[Dict], torch.Tensor],
):
    """Build a vmap'd forward function over population.

    Returns:
        vmap_forward(A_one_dict, B_one_dict, **batch_kwargs) -> output

    The function takes ONE candidate's (A, B) and a batch (without population dim),
    and vmap creates the population dim automatically when called with stacked
    (pop, ...) tensors.

    Usage:
        vfwd = make_vmap_forward(model, state, lambda out: out['cls_repr'])
        # A_pop[name]: (pop, out, rank), B_pop[name]: (pop, rank, in)
        # batch: dict of tensors with NO pop dim
        outputs = vfwd(A_pop, B_pop, **batch)
        # outputs: (pop, batch, ...)
    """
    scale = state.sigma / (state.rank ** 0.5)
    target_names = state.target_param_names
    name_to_safe = state.name_to_safe

    def forward_one(A_one, B_one, **batch_kwargs):
        # Build overrides for ONE candidate
        overrides = {}
        for name in target_names:
            safe = name_to_safe[name]
            M_curr = state.M[safe]  # current learnable value (no grad tracking needed)
            E = A_one[name] @ B_one[name]  # (out, in)
            overrides[name] = M_curr + scale * E
        # functional_call: forward with overridden state_dict
        out = tfunc.functional_call(model, overrides, args=(), kwargs=batch_kwargs)
        return output_extractor(out)

    # Decide vmap in_dims:
    #   First arg (A_one): dict — vmap over dim 0 (pop dim added automatically by stacking)
    #   Second arg (B_one): dict — same
    #   kwargs: broadcast (None)
    # PyTorch vmap supports pytrees for dicts.
    vmap_forward = tfunc.vmap(forward_one, in_dims=(0, 0))
    return vmap_forward


# ----------------------------------------------------------------------
# Chunked vmap call (to fit GPU memory)
# ----------------------------------------------------------------------
@torch.no_grad()
def chunked_vmap_call(
    vmap_forward: Callable,
    A_pop: Dict[str, torch.Tensor],
    B_pop: Dict[str, torch.Tensor],
    batch_kwargs: Dict[str, torch.Tensor],
    chunk_size: int = 8,
) -> torch.Tensor:
    """Call vmap'd forward in chunks to fit memory.

    pop must be divisible by chunk_size (or last chunk smaller).
    Concatenates outputs along dim 0.
    """
    pop = next(iter(A_pop.values())).shape[0]
    if chunk_size >= pop:
        return vmap_forward(A_pop, B_pop, **batch_kwargs)

    chunks = []
    for start in range(0, pop, chunk_size):
        end = min(start + chunk_size, pop)
        A_chunk = {k: v[start:end] for k, v in A_pop.items()}
        B_chunk = {k: v[start:end] for k, v in B_pop.items()}
        out_chunk = vmap_forward(A_chunk, B_chunk, **batch_kwargs)
        chunks.append(out_chunk)
    return torch.cat(chunks, dim=0)


# ----------------------------------------------------------------------
# Fitness shaping (z-score / rank)
# ----------------------------------------------------------------------
def shape_fitness(
    fitnesses: torch.Tensor, kind: str = "zscore"
) -> torch.Tensor:
    if kind == "raw":
        return fitnesses
    if kind == "zscore":
        return (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    if kind == "rank":
        ranks = torch.argsort(torch.argsort(fitnesses)).to(fitnesses.dtype)
        return ranks / (len(fitnesses) - 1) - 0.5
    raise ValueError(f"Unknown fitness shaping: {kind}")


# ----------------------------------------------------------------------
# Ridge regression fitness (full-train, vectorized over population)
# ----------------------------------------------------------------------
@torch.no_grad()
def ridge_fitness_population(
    X_train_pop: torch.Tensor,
    y_train: torch.Tensor,
    X_val_pop: torch.Tensor,
    y_val: torch.Tensor,
    lam: float = 1e-3,
) -> torch.Tensor:
    """Per-candidate ridge regression on cls_repr; returns fitness = -val MSE.

    Args:
        X_train_pop: (pop, N_train, D) per-candidate train embeddings
        y_train:     (N_train,) shared targets
        X_val_pop:   (pop, N_val, D)
        y_val:       (N_val,)
        lam: ridge regularization

    Returns:
        fitnesses: (pop,) — fitness = -val_MSE per candidate
    """
    pop, N_train, D = X_train_pop.shape
    # Augment with bias column
    ones_train = torch.ones(pop, N_train, 1, device=X_train_pop.device, dtype=X_train_pop.dtype)
    Xa_train = torch.cat([X_train_pop, ones_train], dim=-1)  # (pop, N_train, D+1)
    ones_val = torch.ones(pop, X_val_pop.shape[1], 1, device=X_val_pop.device, dtype=X_val_pop.dtype)
    Xa_val = torch.cat([X_val_pop, ones_val], dim=-1)  # (pop, N_val, D+1)

    # Per-candidate: w = (Xa^T Xa + lam I)^-1 Xa^T y
    XtX = torch.einsum("pnd,pne->pde", Xa_train, Xa_train)  # (pop, D+1, D+1)
    eye = torch.eye(D + 1, device=X_train_pop.device, dtype=X_train_pop.dtype).unsqueeze(0)
    A_solve = XtX + lam * eye  # (pop, D+1, D+1)
    Xty = torch.einsum("pnd,n->pd", Xa_train, y_train)  # (pop, D+1)
    w = torch.linalg.solve(A_solve, Xty.unsqueeze(-1)).squeeze(-1)  # (pop, D+1)

    # Predict
    pred_val = torch.einsum("pnd,pd->pn", Xa_val, w)  # (pop, N_val)
    mse_val = torch.mean((pred_val - y_val.unsqueeze(0)) ** 2, dim=1)  # (pop,)
    return -mse_val  # fitness = -MSE
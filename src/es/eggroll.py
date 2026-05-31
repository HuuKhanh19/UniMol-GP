"""
EGGROLL / Evolution-Strategies optimiser for the LoRA subspace (Step 2.2).

We evolve only the LoRA A/B matrices (+ regression head) of a *frozen* UniMol
backbone. Because that search space is already low-rank (LoRA), we perturb those
small matrices with plain antithetic Gaussian noise -- this is exactly the limit
that EGGROLL's low-rank perturbations converge to (paper Thm 3/4), and it keeps
the implementation simple and robust. The population-batched LoRA forward is what
gives EGGROLL its efficiency; here we evaluate members sequentially (correctness
first), which is fine because the frozen-backbone forward on ESOL-sized batches is
cheap. Swap in a vmapped/einsum batched forward later if you need more throughput.

Update (mirrors OpenES / paper recipe):
    * antithetic pairs (+eps, -eps)
    * centered-rank fitness shaping  (robust to RMSE outliers)
    * AdamW outer optimiser on the flat mean vector theta
    * fitness = -loss  (we maximise fitness == minimise MSE)

The optimiser owns ``theta`` (the mean of the trainable params) as a single fp32
leaf so that ``torch.optim.AdamW`` can drive it. ``step()`` scatters perturbed
copies into the live model and calls a user ``forward_loss`` closure that forwards
the model on a fixed minibatch and returns the scalar loss.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn


def _centered_rank(x: torch.Tensor) -> torch.Tensor:
    """Map values to centered ranks in [-0.5, 0.5] (higher value -> higher utility)."""
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    ranks = torch.argsort(torch.argsort(x)).to(x.dtype)
    return ranks / (n - 1) - 0.5


class EggrollES:
    def __init__(
        self,
        params: List[nn.Parameter],
        sigma: float = 1e-2,
        lr: float = 1e-3,
        popsize: int = 256,
        weight_decay: float = 0.0,
        rank_transform: bool = True,
        device: str = "cuda",
        seed: int = 42,
        lr_decay: float = 1.0,        # final LR as a fraction of initial; 1.0 = no decay (constant)
        total_steps: int = None,      # T_max for cosine annealing (= es_steps)
    ):
        assert popsize % 2 == 0, "popsize must be even (antithetic pairs)."
        self.params = list(params)
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.d = int(sum(self.numels))
        self.sigma = float(sigma)
        self.popsize = int(popsize)
        self.rank_transform = bool(rank_transform)
        self.device = device

        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))

        theta0 = torch.cat([p.detach().reshape(-1).float() for p in self.params]).to(device)
        self.mean = nn.Parameter(theta0.clone())              # AdamW drives this leaf
        self.optimizer = torch.optim.AdamW([self.mean], lr=lr, weight_decay=weight_decay)

        # Cosine LR decay: lr(t) anneals from `lr` (t=0) to `lr * lr_decay` (t=total_steps).
        self.scheduler = None
        if lr_decay < 1.0 and total_steps and total_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(total_steps), eta_min=float(lr) * float(lr_decay))

    # ---- helpers -----------------------------------------------------------
    @torch.no_grad()
    def _scatter(self, flat: torch.Tensor) -> None:
        i = 0
        for p, n, s in zip(self.params, self.numels, self.shapes):
            p.copy_(flat[i:i + n].view(s).to(p.dtype))
            i += n

    @torch.no_grad()
    def sync_model(self) -> None:
        """Write the current mean into the live model params."""
        self._scatter(self.mean.detach())

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ---- one ES update on a fixed minibatch --------------------------------
    @torch.no_grad()
    def step(self, forward_loss: Callable[[], float]) -> float:
        """One antithetic ES update. ``forward_loss()`` forwards the model (params
        already set by this method) on the current minibatch and returns scalar loss."""
        half = self.popsize // 2
        eps = torch.randn(half, self.d, generator=self.gen, device=self.device)
        mean = self.mean.detach()

        fitness = torch.empty(self.popsize, device=self.device)
        for k in range(half):
            self._scatter(mean + self.sigma * eps[k]);  fitness[2 * k]     = -float(forward_loss())
            self._scatter(mean - self.sigma * eps[k]);  fitness[2 * k + 1] = -float(forward_loss())

        if self.rank_transform:
            util = _centered_rank(fitness)
        else:
            util = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        u_plus, u_minus = util[0::2], util[1::2]                 # utilities of +eps / -eps
        # ES gradient estimate of the *fitness* (ascent direction).
        grad_fit = (eps * (u_plus - u_minus).unsqueeze(1)).sum(0) / (self.popsize * self.sigma)

        # AdamW minimises -> grad = -grad_fit  (so we ascend fitness == descend loss).
        self.mean.grad = (-grad_fit).detach()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()                               # cosine LR decay

        self.sync_model()                                       # leave model at the new mean
        mean_loss = float((-fitness).mean())                    # mean MSE over the population
        return mean_loss
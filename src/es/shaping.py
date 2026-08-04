"""Fitness shaping for the ES update.

The default (``zscore``) is the paper's GSM8K scoring recipe transplanted to
molecular regression. There, a noise direction is scored by z-normalising its
accuracy *per question* against a global variance and averaging over questions,
so that no single hard question dominates the ranking of perturbations. The
molecular analogue normalises the out-of-fold squared error *per molecule*
across the population: every molecule contributes equally to deciding which
perturbation was good, instead of the handful of high-error outliers that
dominate a plain RMSE.

Antithetic centring is applied on top. Members ``i`` and ``i + N/2`` carry
mirrored perturbations, so subtracting the pair mean removes everything the
pair has in common and leaves only the part attributable to the perturbation
direction.
"""

from __future__ import annotations

import torch

#: Guard for the global standard deviation when a population is degenerate.
_EPS = 1e-12


def per_molecule_z(sq_err: torch.Tensor) -> torch.Tensor:
    """``(N,)`` fitness from ``(N, n_mol)`` out-of-fold squared errors.

    Centres each molecule's errors across the population and divides by the
    *global* spread -- per-molecule scaling would amplify molecules on which the
    population happens to agree, turning numerical noise into signal.
    Higher is better, so the mean z-score is negated.
    """
    err = sq_err.double()
    centred = err - err.mean(dim=0, keepdim=True)
    return (-centred.mean(dim=1) / centred.std().clamp_min(_EPS)).float()


def negative_score(score: torch.Tensor) -> torch.Tensor:
    """Plain fitness from a scalar per-member error (lower error is better)."""
    return -score.double().float()


def centered_rank(fitness: torch.Tensor) -> torch.Tensor:
    """Map fitness to evenly spaced ranks in ``[-0.5, 0.5]``.

    Salimans-style rank shaping: makes the update invariant to monotone
    transforms of the objective and caps the influence of a single outlier
    member, which matters at the population sizes affordable here.
    """
    n = fitness.numel()
    if n < 2:
        return torch.zeros_like(fitness)
    order = fitness.argsort()
    ranks = torch.empty_like(fitness)
    ranks[order] = torch.arange(n, device=fitness.device, dtype=fitness.dtype)
    return ranks / (n - 1) - 0.5


def antithetic_center(fitness: torch.Tensor) -> torch.Tensor:
    """Replace each mirrored pair by its half-difference.

    With ``E`` and ``-E`` evaluated as members ``i`` and ``i + N/2``, the
    estimator only needs the contrast between them; the shared component is pure
    noise for the gradient and dropping it halves the variance.
    """
    n = fitness.numel()
    if n % 2:
        return fitness
    half = n // 2
    delta = (fitness[:half] - fitness[half:]) / 2
    return torch.cat([delta, -delta])


def shape(sq_err: torch.Tensor, mode: str = 'zscore',
          antithetic: bool = True) -> torch.Tensor:
    """Turn ``(N, n_mol)`` squared errors into an ``(N,)`` ES fitness.

    Args:
        mode: ``zscore`` (per-molecule normalisation), ``rank`` (centred rank of
            the scalar RMSE), or ``zscore_rank`` (rank of the z-scored fitness).
        antithetic: subtract the mirrored pair mean afterwards.
    """
    if mode == 'zscore':
        fitness = per_molecule_z(sq_err)
    elif mode == 'rank':
        fitness = centered_rank(negative_score(sq_err.mean(dim=1).sqrt()))
    elif mode == 'zscore_rank':
        fitness = centered_rank(per_molecule_z(sq_err))
    else:
        raise ValueError(f'unknown shaping mode: {mode}')

    fitness = torch.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
    return antithetic_center(fitness) if antithetic else fitness

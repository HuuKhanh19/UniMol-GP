"""Batched ridge regression and grouped cross-validated scoring.

The merge step of the head is ridge over the ``k`` tree outputs plus a linear
probe and an intercept. Because it is closed form, it is solved *exactly* for
every candidate rather than searched -- a variable-projection (VarPro) setup
that removes ``k + 2`` nuisance dimensions from both the GP search and the ES
search, and makes the fitness invariant to any rescaling of the tree outputs.

Every routine here is batched over a leading dimension so that a whole GP
island (candidates) or a whole ES population (perturbed backbones) is scored in
one call.

Fitness is *scaffold-grouped* K-fold CV inside the training split. Plain LOOCV
would be cheaper still -- ridge has a closed-form PRESS -- but it cannot respect
scaffold groups, so it would reward memorising scaffolds, which is exactly what
the outer Bemis-Murcko split penalises.
"""

from __future__ import annotations

import numpy as np
import torch

#: Returned in place of a score when a design matrix is unusable.
BAD_SCORE = float('inf')


def penalty_vector(k: int, alpha: float, n_unpenalized: int = 1,
                   scales: dict[int, float] | None = None) -> np.ndarray:
    """Per-column ridge penalties.

    The intercept (and any other leading column the caller wants free) is left
    unpenalised, which is the standard convention: shrinking the intercept makes
    the fit depend on where the target happens to be centred. ``scales`` lets a
    specific column be penalised harder or softer than the rest -- used to keep
    the linear probe from becoming a crutch that starves the trees.
    """
    diag = np.full(k, alpha, dtype=np.float64)
    diag[:n_unpenalized] = 0.0
    for col, factor in (scales or {}).items():
        diag[col] *= factor
    return diag


def solve(gram: torch.Tensor, rhs: torch.Tensor,
          penalty: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Solve ``(G + diag(penalty)) beta = rhs``, batched over the leading dim.

    Falls back to ``lstsq`` for any batch whose system is singular, so a
    degenerate candidate produces a finite (if useless) coefficient vector
    instead of raising and killing the generation.
    """
    a = gram.double()
    diag = torch.as_tensor(penalty, device=a.device, dtype=a.dtype)
    a = a + torch.diag(diag)
    b = rhs.double().unsqueeze(-1)
    try:
        beta = torch.linalg.solve(a, b)
    except Exception:
        beta = torch.linalg.lstsq(a, b).solution
    return beta.squeeze(-1)


def _grams(phi: torch.Tensor, y: torch.Tensor,
           rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(Phi^T Phi, Phi^T y)`` over ``rows``, batched over designs."""
    sub = phi.index_select(1, rows)          # (N, n_sub, k)
    ysub = y.index_select(0, rows)           # (n_sub,)
    gram = torch.einsum('nik,nil->nkl', sub, sub)
    rhs = torch.einsum('nik,i->nk', sub, ysub)
    return gram, rhs


def cv_predict(phi: torch.Tensor, y: torch.Tensor, folds: list[torch.Tensor],
               penalty: np.ndarray) -> torch.Tensor:
    """Out-of-fold predictions for a batch of design matrices.

    Args:
        phi: ``(N, n, k)`` design matrices -- N candidates/members sharing rows.
        y: ``(n,)`` targets.
        folds: per-fold row indices; each row appears in exactly one fold.
        penalty: ``(k,)`` per-column ridge penalties, see ``penalty_vector``.

    Returns:
        ``(N, n)`` predictions, each row predicted by a model that never saw it.
    """
    n_designs, n_rows, _ = phi.shape
    out = torch.zeros(n_designs, n_rows, device=phi.device, dtype=torch.float64)
    all_rows = torch.arange(n_rows, device=phi.device)

    for held in folds:
        mask = torch.ones(n_rows, dtype=torch.bool, device=phi.device)
        mask[held] = False
        gram, rhs = _grams(phi, y, all_rows[mask])
        beta = solve(gram, rhs, penalty)                       # (N, k)
        out[:, held] = torch.einsum(
            'nik,nk->ni', phi.index_select(1, held).double(), beta
        )
    return out


def cv_squared_error(phi: torch.Tensor, y: torch.Tensor,
                     folds: list[torch.Tensor], penalty: np.ndarray
                     ) -> torch.Tensor:
    """``(N, n)`` per-row out-of-fold squared error.

    Kept per-row rather than reduced because ES fitness shaping z-scores errors
    *per molecule* across the population -- the molecular analogue of the
    per-question normalisation the paper uses for GSM8K.
    """
    pred = cv_predict(phi, y, folds, penalty)
    return (pred - y.double()).pow(2)


def cv_score(phi: torch.Tensor, y: torch.Tensor, folds: list[torch.Tensor],
             penalty: np.ndarray, invalid: torch.Tensor | None = None,
             ) -> torch.Tensor:
    """``(N,)`` grouped K-fold CV RMSE. ``invalid`` designs score ``inf``."""
    score = cv_squared_error(phi, y, folds, penalty).mean(dim=1).sqrt().float()
    score = torch.nan_to_num(score, nan=BAD_SCORE, posinf=BAD_SCORE)
    if invalid is not None:
        score = torch.where(invalid, torch.full_like(score, BAD_SCORE), score)
    return score


def fit(phi: torch.Tensor, y: torch.Tensor, penalty: np.ndarray) -> torch.Tensor:
    """Fit ridge on all rows. ``phi`` is ``(N, n, k)``; returns ``(N, k)``."""
    rows = torch.arange(phi.shape[1], device=phi.device)
    gram, rhs = _grams(phi, y, rows)
    return solve(gram, rhs, penalty)


def predict(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``(N, n, k) x (N, k) -> (N, n)``."""
    return torch.einsum('nik,nk->ni', phi.double(), beta.double()).float()


def rmse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row-wise RMSE of ``(N, n)`` predictions against ``(n,)`` targets."""
    return (pred - y).pow(2).mean(dim=-1).sqrt()


# --- fold construction -------------------------------------------------------


def scaffold_folds(scaffolds: list[str], n_folds: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Assign rows to folds so that no Murcko scaffold spans two folds.

    Groups are placed largest-first into whichever fold is currently smallest,
    which keeps folds close to equal size even when a few scaffolds dominate.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)

    keys = [k for k in np.array(list(groups), dtype=object)[
        rng.permutation(len(groups))]]
    keys.sort(key=lambda k: -len(groups[k]))

    fold_of = np.zeros(len(scaffolds), dtype=np.int64)
    load = np.zeros(n_folds, dtype=np.int64)
    for key in keys:
        f = int(load.argmin())
        for i in groups[key]:
            fold_of[i] = f
        load[f] += len(groups[key])
    return fold_of


def fold_indices(fold_of: np.ndarray, n_folds: int,
                 device: torch.device) -> list[torch.Tensor]:
    """Turn a fold-id array into per-fold row-index tensors."""
    return [
        torch.from_numpy(np.nonzero(fold_of == f)[0]).to(device)
        for f in range(n_folds)
    ]


def select_rho(phi: torch.Tensor, y: torch.Tensor, folds: list[torch.Tensor],
               rhos: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
               n_unpenalized: int = 1,
               scales: dict[int, float] | None = None) -> tuple[float, float]:
    """Pick the ridge penalty by grouped CV on a single design matrix.

    ``rho`` is expressed relative to ``n_rows``: the tree columns are unit-RMS,
    so the Gram diagonal is about ``n_rows`` and ``alpha = rho * n_rows`` makes
    the same ``rho`` mean the same thing at any dataset size.

    Returns ``(rho, score)`` for the best rho.
    """
    n_rows, k = phi.shape[1], phi.shape[2]
    best = (rhos[0], BAD_SCORE)
    for rho in rhos:
        pen = penalty_vector(k, rho * n_rows, n_unpenalized, scales)
        score = float(cv_score(phi, y, folds, pen)[0])
        if score < best[1]:
            best = (rho, score)
    return best

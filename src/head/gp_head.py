"""The symbolic head: k formula trees over disjoint embedding regions, merged
by ridge.

Design matrix column order is fixed everywhere in the project:

    0                intercept
    1                linear probe   (optional, ``use_probe``)
    2 .. 2 + k - 1   tree outputs

The probe is a single scalar column ``w_lin . z + b_lin``, not the raw 512
dimensions. Feeding all 512 columns to ridge would make the per-member solve
``O(512^3)`` and dominate the backbone forward pass by several times; one probe
column costs nothing and still guarantees the head can reproduce the linear
baseline exactly, which is the point of having it.
"""

from __future__ import annotations

import numpy as np
import torch

from src.head.evaluator import DEGENERATE_STD, evaluate
from src.head.genome import Population, to_infix

#: Column index of the intercept, and how many leading columns ridge leaves
#: unpenalised.
INTERCEPT_COL = 0
N_UNPENALIZED = 1


def contiguous_regions(embed_dim: int, n_trees: int) -> list[np.ndarray]:
    """Split ``[0, embed_dim)`` into ``n_trees`` disjoint contiguous blocks.

    Disjointness is the point: it forces the trees apart, and near-collinear
    columns are exactly what makes the ridge merge unstable.
    """
    if embed_dim % n_trees:
        raise ValueError(f'{embed_dim} not divisible by n_trees={n_trees}')
    width = embed_dim // n_trees
    return [np.arange(j * width, (j + 1) * width) for j in range(n_trees)]


def random_regions(embed_dim: int, n_trees: int, width: int,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """Overlapping random subspaces, one per tree (random-subspace ablation)."""
    return [
        np.sort(rng.choice(embed_dim, size=width, replace=False))
        for _ in range(n_trees)
    ]


class GPHead:
    """k symbolic trees plus a ridge merge over their outputs.

    Args:
        genomes: ``k`` postfix trees, one per region.
        regions: global embedding columns each tree may read.
        use_probe: include the linear probe column.
    """

    def __init__(self, genomes: Population, regions: list[np.ndarray],
                 use_probe: bool = True):
        self.genomes = genomes
        self.regions = regions
        self.use_probe = use_probe
        self.beta: np.ndarray | None = None
        self.probe_w: np.ndarray | None = None
        self.probe_b: float = 0.0
        #: One probe per CV fold, each fitted without that fold. Used only while
        #: searching; the deployed head uses the all-rows probe above.
        self.probe_folds_w: np.ndarray | None = None
        self.probe_folds_b: np.ndarray | None = None
        #: Frozen normalisation captured at fit time. While searching we
        #: recompute RMS per batch (per ES member, even) because that scale
        #: invariance is part of what VarPro buys us; once the head is frozen
        #: the training scale must travel with it or test predictions shift.
        self.col_rms: np.ndarray | None = None
        self.probe_rms: float = 1.0

    @property
    def n_trees(self) -> int:
        return self.genomes.size

    @property
    def n_cols(self) -> int:
        return 1 + int(self.use_probe) + self.n_trees

    @property
    def tree_col_offset(self) -> int:
        return 1 + int(self.use_probe)

    # --- design matrix -------------------------------------------------------

    def probe_column(self, z: torch.Tensor,
                     fold_of: np.ndarray | None = None) -> torch.Tensor:
        """``(M, n, 1)`` linear-probe feature.

        With ``fold_of``, each row is scored by the probe fitted *without* that
        row's CV fold. This matters: the deployment probe is fitted on every
        training row, so using it as a column inside a CV over those same rows
        leaks the labels -- the column already encodes ``y`` for every held-out
        molecule, the CV score collapses, and the trees look worthless because
        they are competing against a fitted prediction rather than against raw
        features.
        """
        if fold_of is None or self.probe_folds_w is None:
            w = torch.as_tensor(self.probe_w, device=z.device, dtype=z.dtype)
            col = z @ w + self.probe_b
        else:
            w = torch.as_tensor(self.probe_folds_w, device=z.device, dtype=z.dtype)
            b = torch.as_tensor(self.probe_folds_b, device=z.device, dtype=z.dtype)
            f = torch.as_tensor(fold_of, device=z.device, dtype=torch.long)
            col = (z * w[f]).sum(dim=-1) + b[f]
        return col.unsqueeze(-1) / max(self.probe_rms, 1e-12)

    def tree_columns(self, z: torch.Tensor) -> torch.Tensor:
        """``(M, n, k)`` raw tree outputs for ``z`` of shape ``(..., n, d)``."""
        n, d = z.shape[-2], z.shape[-1]
        cols = evaluate(self.genomes, z.reshape(-1, d))   # (k, M*n)
        return cols.reshape(self.n_trees, -1, n).permute(1, 2, 0)

    def design(self, z: torch.Tensor, freeze_scale: bool = False,
               fold_of: np.ndarray | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the ridge design matrix.

        Args:
            z: ``(n, d)`` or ``(M, n, d)`` embeddings.
            freeze_scale: use the stored ``col_rms`` instead of the batch RMS.
            fold_of: ``(n,)`` CV fold per row. Pass it whenever the design feeds
                a cross-validated score, so the probe column is out-of-fold.

        Returns:
            ``(design, rms)`` where design is ``(..., n, n_cols)`` and rms is
            ``(M, 1, k)`` -- the scale actually applied, so a caller that is
            fitting a final head can store it.
        """
        lead = tuple(z.shape[:-2])
        n = z.shape[-2]
        cols = self.tree_columns(z)                       # (M, n, k)

        if freeze_scale and self.col_rms is not None:
            rms = torch.as_tensor(
                self.col_rms, device=z.device, dtype=cols.dtype
            ).view(1, 1, -1).expand(cols.shape[0], 1, -1)
        else:
            rms = cols.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        cols = cols / rms

        parts = [torch.ones(cols.shape[0], n, 1, device=z.device, dtype=cols.dtype)]
        if self.use_probe:
            parts.append(
                self.probe_column(z.reshape(-1, n, z.shape[-1]), fold_of)
            )
        parts.append(cols)
        design = torch.cat(parts, dim=-1)
        return design.reshape(*lead, n, self.n_cols) if lead else design[0], rms

    def degenerate(self, z: torch.Tensor) -> torch.Tensor:
        """``(M, k)`` bool: trees that emit a constant and carry no signal."""
        return self.tree_columns(z).std(dim=1) < DEGENERATE_STD

    # --- prediction ----------------------------------------------------------

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """``(n,)`` predictions from the frozen head."""
        if self.beta is None:
            raise RuntimeError('GPHead.beta is unset -- call fit_final() first')
        design, _ = self.design(z, freeze_scale=True)
        if design.dim() == 2:
            design = design.unsqueeze(0)
        beta = torch.as_tensor(self.beta, device=z.device, dtype=torch.float64)
        return torch.einsum('mnk,k->mn', design.double(), beta).squeeze(0).float()

    def contributions(self, z: torch.Tensor) -> dict[str, float]:
        """Share of prediction variance carried by trees vs the linear probe.

        Worth logging every phase: with ``use_probe`` on it is entirely possible
        for ridge to lean on the probe and leave the trees vestigial, and that
        failure is invisible in the RMSE alone.
        """
        design, _ = self.design(z, freeze_scale=True)
        if design.dim() == 2:
            design = design.unsqueeze(0)
        beta = torch.as_tensor(self.beta, device=z.device, dtype=design.dtype)
        terms = design[0] * beta                                    # (n, n_cols)
        var = terms.var(dim=0)
        total = float(var.sum()) or 1.0
        off = self.tree_col_offset
        out = {'trees': float(var[off:].sum()) / total}
        if self.use_probe:
            out['probe'] = float(var[1]) / total
        return out

    # --- reporting / persistence --------------------------------------------

    def formulas(self, precision: int = 3) -> list[str]:
        """Infix source of every tree, for the paper's interpretability table."""
        return [
            to_infix(self.genomes, j, precision=precision)
            for j in range(self.n_trees)
        ]

    def sizes(self) -> np.ndarray:
        return self.genomes.length.copy()

    def save(self, path: str) -> None:
        np.savez(
            path,
            code=self.genomes.code, arg=self.genomes.arg,
            const=self.genomes.const, length=self.genomes.length,
            regions=np.stack(self.regions),
            beta=np.asarray(self.beta if self.beta is not None else []),
            probe_w=np.asarray(self.probe_w if self.probe_w is not None else []),
            probe_b=np.asarray(self.probe_b),
            probe_folds_w=np.asarray(
                self.probe_folds_w if self.probe_folds_w is not None else []),
            probe_folds_b=np.asarray(
                self.probe_folds_b if self.probe_folds_b is not None else []),
            col_rms=np.asarray(self.col_rms if self.col_rms is not None else []),
            probe_rms=np.asarray(self.probe_rms),
            use_probe=np.asarray(self.use_probe),
        )

    @classmethod
    def load(cls, path: str) -> 'GPHead':
        d = np.load(path, allow_pickle=False)
        head = cls(
            Population(d['code'], d['arg'], d['const'], d['length']),
            [row for row in d['regions']],
            use_probe=bool(d['use_probe']),
        )
        head.beta = d['beta'] if d['beta'].size else None
        head.probe_w = d['probe_w'] if d['probe_w'].size else None
        head.probe_b = float(d['probe_b'])
        head.probe_folds_w = (
            d['probe_folds_w'] if d['probe_folds_w'].size else None)
        head.probe_folds_b = (
            d['probe_folds_b'] if d['probe_folds_b'].size else None)
        head.col_rms = d['col_rms'] if d['col_rms'].size else None
        head.probe_rms = float(d['probe_rms'])
        return head


def fit_probe(z: torch.Tensor, y: torch.Tensor, rho: float = 1e-2
              ) -> tuple[np.ndarray, float, float]:
    """Ridge linear probe on the raw embedding: ``y ~ w . z + b``.

    Refit once per GP phase, while the backbone is frozen, then held fixed for
    the whole ES phase -- otherwise the probe column would be chasing a moving
    representation and the ES fitness would stop being a clean function of the
    perturbation.

    Returns ``(w, b, rms)`` where ``rms`` is the scale of the resulting column.
    """
    zc = z.double()
    n, d = zc.shape
    mean = zc.mean(dim=0, keepdim=True)
    zc = zc - mean
    gram = zc.t() @ zc + rho * n * torch.eye(d, device=z.device, dtype=zc.dtype)
    yc = y.double()
    w = torch.linalg.solve(gram, zc.t() @ (yc - yc.mean()))
    b = float(yc.mean() - (mean @ w).squeeze())
    col = (z.double() @ w + b)
    rms = float(col.pow(2).mean().sqrt().clamp_min(1e-12))
    return w.float().cpu().numpy(), b, rms


def fit_probe_folds(z: torch.Tensor, y: torch.Tensor, fold_of: np.ndarray,
                    n_folds: int, rho: float = 1e-2
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """One linear probe per CV fold, each fitted on the *other* folds.

    Assembling the probe column from these makes it out-of-fold for any subset
    of the training rows, which is what the GP and ES fitness both need: a probe
    fitted on all rows would hand the cross-validation a column that already
    encodes the held-out labels.

    Returns ``(W, b, rms)`` with ``W`` of shape ``(n_folds, d)``.
    """
    ws, bs = [], []
    for f in range(n_folds):
        keep = np.nonzero(fold_of != f)[0]
        idx = torch.as_tensor(keep, device=z.device, dtype=torch.long)
        w, b, _ = fit_probe(z.index_select(0, idx), y.index_select(0, idx), rho)
        ws.append(w)
        bs.append(b)

    w_all = np.stack(ws).astype(np.float32)
    b_all = np.asarray(bs, dtype=np.float32)
    wt = torch.as_tensor(w_all, device=z.device, dtype=z.dtype)
    bt = torch.as_tensor(b_all, device=z.device, dtype=z.dtype)
    ft = torch.as_tensor(fold_of, device=z.device, dtype=torch.long)
    col = (z * wt[ft]).sum(dim=-1) + bt[ft]
    return w_all, b_all, float(col.pow(2).mean().sqrt().clamp_min(1e-12))

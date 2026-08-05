"""EGGROLL training loop for the UniMol backbone under a fixed symbolic head.

One step:

1. draw a fold-stratified molecule batch (refreshed every ``batch_refresh``
   steps -- holding it fixed across steps is a common-random-numbers device that
   keeps consecutive gradient estimates comparable);
2. run the frozen prefix once and share it across the whole population;
3. for each population chunk, run the perturbed suffix and score the members by
   grouped-CV ridge error under the *same* trees, the *same* folds and the
   *same* ridge penalty, so fitness differences come only from the perturbation;
4. shape the per-molecule errors into a fitness and take one Adam step on the
   aggregated low-rank update.

The ridge weights are re-solved exactly for every member rather than searched.
That is variable projection: it removes the merge weights from the search space
and makes fitness invariant to any rescaling of the tree outputs, which is a
real reduction in the effective dimension ES has to cover -- not just a saving.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from src.es import perturb, shaping
from src.es.data import MoleculeData, stratified_batch
from src.es.forward_unimol import SplitUniMol
from src.head import ridge
from src.head.gp_head import N_UNPENALIZED, GPHead


@torch.no_grad()
def embed(split: SplitUniMol, data: MoleculeData, idx: np.ndarray,
          device: torch.device, mol_tile: int, pert=None,
          n_members: int = 1) -> torch.Tensor:
    """``(n_members, len(idx), 512)`` CLS representations.

    Molecules go through in length-sorted tiles so each tile pads only to its
    own longest molecule -- attention cost and memory scale with ``S^2``, and a
    random ESOL batch otherwise pads to near the global maximum. Results are
    scattered back into ``idx`` order.
    """
    out = torch.zeros(
        n_members, idx.size, split.embed_dim, device=device, dtype=split.dtype,
    )
    for positions in data.length_tiles(idx, mol_tile):
        batch = data.collate(idx[positions], device)
        x0, bias0 = split.prefix(batch)
        z = split.suffix(x0, bias0, n_members, pert)
        out[:, torch.from_numpy(positions).to(device), :] = z
    return out.float()


@dataclass
class ESConfig:
    """EGGROLL hyperparameters."""

    steps: int = 2000
    pop_size: int = 256
    #: Members evaluated per forward pass. Peak memory is ~3 attention-score
    #: tensors of ``pop_chunk * mol_tile * heads * S^2``; on a 16 GB card this
    #: is the first knob to turn down.
    pop_chunk: int = 16
    #: Molecules scored per ES step. The CV fitness is computed within this
    #: batch, so it also sets how noisy the fitness is.
    mol_batch: int = 256
    mol_tile: int = 64
    #: Steps between resampling the molecule batch.
    batch_refresh: int = 1
    sigma: float = 3e-3
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    shaping: str = 'zscore'
    antithetic: bool = True
    n_folds: int = 5
    rho: float = 1e-2
    probe_penalty_scale: float = 1.0
    eval_every: int = 25
    seed: int = 0


class EGGROLL:
    """ES over the UniMol suffix, with the symbolic head held fixed."""

    def __init__(self, split: SplitUniMol, data: MoleculeData,
                 fold_of: np.ndarray, cfg: ESConfig, device: torch.device):
        self.split = split
        self.data = data
        self.fold_of = fold_of
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(cfg.seed)
        self.generator = torch.Generator(device=device)
        self.step_idx = 0

        self.head: GPHead | None = None
        self.rho = cfg.rho
        self.optimizer = perturb.RelativeAdam(
            split.targets, split.shapes(), split.sigma_scale, device,
            lr=cfg.lr, betas=cfg.betas,
        )
        self._batch: np.ndarray | None = None
        self._batch_folds: list[torch.Tensor] = []
        self._batch_fold_of: np.ndarray | None = None
        self._batch_y: torch.Tensor | None = None

    # --- wiring --------------------------------------------------------------

    def set_head(self, head: GPHead, rho: float) -> None:
        """Install the head ES optimises against. Fixed for the whole phase."""
        self.head = head
        self.rho = rho

    def _penalty(self, n_rows: int) -> np.ndarray:
        scales = None
        if self.head.use_probe and self.cfg.probe_penalty_scale != 1.0:
            scales = {1: self.cfg.probe_penalty_scale}
        return ridge.penalty_vector(
            self.head.n_cols, self.rho * n_rows, N_UNPENALIZED, scales
        )

    def _refresh_batch(self) -> None:
        cfg = self.cfg
        self._batch = stratified_batch(
            self.fold_of, cfg.n_folds, cfg.mol_batch, self.rng
        )
        folds_here = self.fold_of[self._batch]
        self._batch_folds = [
            torch.from_numpy(np.nonzero(folds_here == f)[0]).to(self.device)
            for f in range(cfg.n_folds)
            if np.any(folds_here == f)
        ]
        self._batch_fold_of = folds_here
        self._batch_y = self.data.targets(self._batch, self.device)

    # --- forward -------------------------------------------------------------

    def embed(self, idx: np.ndarray, pert=None, n_members: int = 1
              ) -> torch.Tensor:
        return embed(self.split, self.data, idx, self.device,
                     self.cfg.mol_tile, pert, n_members)

    @torch.no_grad()
    def squared_error(self, z: torch.Tensor, y: torch.Tensor,
                      folds: list[torch.Tensor],
                      fold_of: np.ndarray) -> torch.Tensor:
        """``(N, n)`` out-of-fold squared error for each population member.

        ``fold_of`` keeps the probe column out-of-fold; without it the column
        carries labels for the very rows the CV holds out, the score collapses
        and every perturbation looks equally good.
        """
        design, _ = self.head.design(z, fold_of=fold_of)
        if design.dim() == 2:
            design = design.unsqueeze(0)
        return ridge.cv_squared_error(design, y, folds, self._penalty(z.shape[-2]))

    # --- one ES step ---------------------------------------------------------

    def step(self) -> dict:
        cfg = self.cfg
        if self.head is None:
            raise RuntimeError('call set_head() before stepping')
        if self._batch is None or self.step_idx % cfg.batch_refresh == 0:
            self._refresh_batch()

        t0 = time.time()
        self.generator.manual_seed(cfg.seed * 1_000_003 + self.step_idx)
        pert = perturb.sample(
            self.split, cfg.pop_size, cfg.sigma, self.generator, cfg.antithetic
        )

        errors = []
        for start in range(0, cfg.pop_size, cfg.pop_chunk):
            stop = min(start + cfg.pop_chunk, cfg.pop_size)
            z = self.embed(self._batch, pert.slice(start, stop), stop - start)
            errors.append(self.squared_error(
                z, self._batch_y, self._batch_folds, self._batch_fold_of))
        sq = torch.cat(errors, dim=0)                        # (pop, n_batch)

        fitness = shaping.shape(sq, cfg.shaping, cfg.antithetic)
        grads: dict = {}
        perturb.accumulate(pert, fitness, grads)
        for key in grads:
            grads[key] /= cfg.pop_size
        magnitude = self.optimizer.step(self.split, grads)

        rmse = sq.mean(dim=1).sqrt()
        self.step_idx += 1
        return {
            'step': self.step_idx,
            'pop_cv_rmse_mean': float(rmse.mean()),
            'pop_cv_rmse_best': float(rmse.min()),
            'update_rel': magnitude,
            'sec': time.time() - t0,
        }

    # --- unperturbed diagnostics --------------------------------------------

    @torch.no_grad()
    def centre_score(self, idx: np.ndarray | None = None) -> float:
        """Grouped-CV RMSE of the *unperturbed* mean model, in scaled units.

        This -- not the population mean -- is the quantity ES is trying to move,
        and it is the honest signal for whether a phase helped.
        """
        idx = self._batch if idx is None else idx
        folds_here = self.fold_of[idx]
        folds = [
            torch.from_numpy(np.nonzero(folds_here == f)[0]).to(self.device)
            for f in range(self.cfg.n_folds)
            if np.any(folds_here == f)
        ]
        z = self.embed(idx, None, 1)
        y = self.data.targets(idx, self.device)
        return float(self.squared_error(z, y, folds, folds_here).mean().sqrt())

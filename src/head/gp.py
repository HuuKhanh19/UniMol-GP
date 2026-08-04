"""Cooperative coevolutionary GP for the k-tree head.

Each tree gets its own sub-population ("island") on its own embedding region.
A candidate for slot ``j`` is scored *in context*: drop it into slot ``j``, keep
the other ``k - 1`` slots at their current representatives, re-solve ridge, take
the grouped-CV RMSE. That is Potter & De Jong cooperative coevolution, and it is
the right decomposition here because the merge is linear in the tree outputs --
so scoring a whole island is one batched solve rather than one fit per candidate.

Two things keep this from overfitting 900 molecules with 512 features on tap:
the fitness is scaffold-grouped CV (never training error), and search is
restricted per tree to a 32-column region.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.head import ridge
from src.head.evaluator import evaluate, normalise
from src.head.genome import (
    Population,
    concat,
    crossover,
    max_len_for_depth,
    mutate,
    random_population,
)
from src.head.gp_head import (
    GPHead,
    N_UNPENALIZED,
    contiguous_regions,
    fit_probe,
    random_regions,
)


@dataclass
class GPConfig:
    """Search settings for the symbolic head."""

    n_trees: int = 16
    pop_size: int = 200
    max_depth: int = 5
    min_depth: int = 2
    tournament: int = 5
    elite: int = 2
    p_crossover: float = 0.8
    p_point: float = 0.15
    p_subtree: float = 0.10
    p_const: float = 0.15
    const_sigma: float = 0.3
    #: Size penalty added to the CV RMSE, per node. At RMSE ~0.7 and ~30 nodes,
    #: 5e-4 costs a tree about 0.015 -- enough to break ties toward the smaller
    #: formula without letting size override accuracy.
    parsimony: float = 5e-4
    use_probe: bool = True
    #: Extra shrinkage on the probe column. >1 makes the linear crutch more
    #: expensive so the trees have to earn their keep.
    probe_penalty_scale: float = 1.0
    region_mode: str = 'contiguous'   # 'contiguous' | 'random'
    region_width: int | None = None   # only for 'random'
    #: Number of perturbed-backbone replicas to average fitness over. 1 = off.
    #: Above 1, trees are selected for robustness across the ES neighbourhood of
    #: the current backbone, at a proportional cost in CV compute.
    replicas: int = 1
    rho: float = 1e-2
    rho_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    tune_rho: bool = True


@dataclass
class GPState:
    """Everything the search carries between phases (warm start)."""

    islands: list[Population] = field(default_factory=list)
    representatives: Population | None = None
    #: Snapshot of the best full head seen in the current phase. Kept separate
    #: because a cooperative representative is only best *in the context it was
    #: scored in* -- once the other islands move, it can get worse.
    best_genomes: Population | None = None
    best_score: float = float('inf')


class CoevolutionGP:
    """Evolve ``k`` trees jointly against a shared ridge merge."""

    def __init__(self, cfg: GPConfig, embed_dim: int, rng: np.random.Generator,
                 device: torch.device):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.embed_dim = embed_dim
        self.max_len = max_len_for_depth(cfg.max_depth)

        if cfg.region_mode == 'random':
            width = cfg.region_width or (embed_dim // cfg.n_trees)
            self.regions = random_regions(embed_dim, cfg.n_trees, width, rng)
        else:
            self.regions = contiguous_regions(embed_dim, cfg.n_trees)

        self.state = GPState(
            islands=[
                random_population(cfg.pop_size, self.regions[j], rng,
                                  max_depth=cfg.max_depth,
                                  min_depth=cfg.min_depth, max_len=self.max_len)
                for j in range(cfg.n_trees)
            ]
        )
        # Representative = first member of each island until the first scoring
        # pass replaces it with that island's in-context best.
        self.state.representatives = concat(
            [self.state.islands[j][0] for j in range(cfg.n_trees)]
        )
        self.probe_w: np.ndarray | None = None
        self.probe_b = 0.0
        self.probe_rms = 1.0
        self.rho = cfg.rho

    # --- helpers -------------------------------------------------------------

    def head(self, best: bool = False) -> GPHead:
        """A GPHead view of the trees (ridge weights not fitted yet).

        ``best=True`` returns the archived best-scoring combination from the
        last phase rather than the live representatives, which is what should be
        handed to ES and to final evaluation.
        """
        genomes = self.state.best_genomes if best and self.state.best_genomes \
            is not None else self.state.representatives
        head = GPHead(genomes.copy(), self.regions, use_probe=self.cfg.use_probe)
        head.probe_w, head.probe_b, head.probe_rms = (
            self.probe_w, self.probe_b, self.probe_rms
        )
        return head

    def _penalty(self, n_cols: int, n_rows: int) -> np.ndarray:
        scales = None
        if self.cfg.use_probe and self.cfg.probe_penalty_scale != 1.0:
            scales = {1: self.cfg.probe_penalty_scale}
        return ridge.penalty_vector(n_cols, self.rho * n_rows, N_UNPENALIZED, scales)

    def _tournament(self, fitness: np.ndarray, n: int) -> np.ndarray:
        idx = self.rng.integers(0, fitness.size, size=(n, self.cfg.tournament))
        return idx[np.arange(n), fitness[idx].argmin(axis=1)]

    def _breed(self, island: Population, fitness: np.ndarray,
               cols: np.ndarray) -> Population:
        cfg = self.cfg
        p = island.size
        kids = island[self._tournament(fitness, p)]
        mates = island[self._tournament(fitness, p)]
        crossed = crossover(kids, mates, self.rng)

        use = self.rng.random(p) < cfg.p_crossover
        for dst, src in ((kids.code, crossed.code), (kids.arg, crossed.arg),
                         (kids.const, crossed.const)):
            dst[use] = src[use]
        kids.length[use] = crossed.length[use]

        kids = mutate(kids, self.rng, cols, p_point=cfg.p_point,
                      p_subtree=cfg.p_subtree, p_const=cfg.p_const,
                      const_sigma=cfg.const_sigma, max_depth=cfg.min_depth + 1)

        if cfg.elite > 0:
            keep = np.argsort(fitness)[: cfg.elite]
            elites = island[keep]
            for dst, src in ((kids.code, elites.code), (kids.arg, elites.arg),
                             (kids.const, elites.const)):
                dst[: cfg.elite] = src
            kids.length[: cfg.elite] = elites.length
        return kids

    # --- main loop -----------------------------------------------------------

    def evolve(self, z: torch.Tensor, y: torch.Tensor,
               folds: list[torch.Tensor], n_gens: int,
               log_every: int = 0) -> list[dict]:
        """Run ``n_gens`` cooperative generations against embeddings ``z``.

        Args:
            z: ``(n, d)`` embeddings, or ``(R, n, d)`` replicas from distinct
                perturbed backbones when ``cfg.replicas > 1``.
            y: ``(n,)`` targets.
            folds: scaffold-grouped row indices over the ``n`` original rows.
            n_gens: generations; each visits all ``k`` islands once.

        Returns:
            One record per logged generation.
        """
        cfg = self.cfg
        if z.dim() == 2:
            z = z.unsqueeze(0)
        n_rep, n_rows, _ = z.shape
        z_flat = z.reshape(-1, z.shape[-1])
        n_eff = z_flat.shape[0]

        # A molecule must land in the same fold in every replica, or a replica
        # of a held-out molecule would leak into the training folds.
        folds_eff = [
            torch.cat([f + r * n_rows for r in range(n_rep)]) for f in folds
        ]

        self.probe_w, self.probe_b, self.probe_rms = (
            fit_probe(z[0], y, rho=cfg.rho) if cfg.use_probe else (None, 0.0, 1.0)
        )
        head = self.head()
        off = head.tree_col_offset
        n_cols = head.n_cols
        y_eff = y.repeat(n_rep)

        base, _ = head.design(z_flat)                      # (n_eff, n_cols)
        if cfg.tune_rho:
            self.rho, _ = ridge.select_rho(
                base.unsqueeze(0), y_eff, folds_eff, cfg.rho_grid, N_UNPENALIZED,
                {1: cfg.probe_penalty_scale} if cfg.use_probe else None,
            )
        penalty = self._penalty(n_cols, n_eff)

        # Scores from a previous phase were measured against a different
        # backbone, so they are not comparable -- restart the archive.
        self.state.best_score = float('inf')
        self.state.best_genomes = None

        history: list[dict] = []
        for gen in range(n_gens):
            t0 = time.time()
            for j in self.rng.permutation(cfg.n_trees):
                island = self.state.islands[j]
                raw = evaluate(island, z_flat)             # (P, n_eff)
                phi, bad = normalise(raw)

                design = base.unsqueeze(0).repeat(island.size, 1, 1)
                design[:, :, off + j] = phi
                score = ridge.cv_score(design, y_eff, folds_eff, penalty, bad)

                fitness = (score + cfg.parsimony * torch.as_tensor(
                    island.length, device=score.device, dtype=score.dtype
                )).cpu().numpy()

                best = int(np.argmin(fitness))
                # Every candidate can be degenerate early on; keeping the old
                # representative is better than adopting a constant tree.
                if not np.isfinite(fitness[best]):
                    self.state.islands[j] = self._breed(
                        island, fitness, self.regions[j])
                    continue
                self._install(j, island[best], phi[best], base, off)
                if float(score[best]) < self.state.best_score:
                    self.state.best_score = float(score[best])
                    self.state.best_genomes = self.state.representatives.copy()

                self.state.islands[j] = self._breed(island, fitness, self.regions[j])

            if log_every and (gen % log_every == 0 or gen == n_gens - 1):
                history.append({
                    'gen': gen,
                    'cv_rmse': self.state.best_score,
                    'mean_size': float(np.mean([
                        float(isl.length.mean()) for isl in self.state.islands
                    ])),
                    'rho': self.rho,
                    'sec': time.time() - t0,
                })
        return history

    def _install(self, j: int, genome: Population, column: torch.Tensor,
                 base: torch.Tensor, offset: int) -> None:
        """Adopt ``genome`` as the representative for island ``j``."""
        reps = self.state.representatives
        for dst, src in ((reps.code, genome.code), (reps.arg, genome.arg),
                         (reps.const, genome.const)):
            dst[j] = src[0]
        reps.length[j] = genome.length[0]
        base[:, offset + j] = column


def fit_final(head: GPHead, z: torch.Tensor, y: torch.Tensor, rho: float,
              probe_penalty_scale: float = 1.0) -> GPHead:
    """Freeze a head: fit ridge on all rows and capture the training scale.

    After this the head is a fixed function -- ``col_rms`` travels with it, so
    test-time columns are scaled exactly as training-time columns were.
    """
    design, rms = head.design(z)
    if design.dim() == 2:
        design = design.unsqueeze(0)
    scales = {1: probe_penalty_scale} if head.use_probe else None
    penalty = ridge.penalty_vector(
        head.n_cols, rho * design.shape[1], N_UNPENALIZED, scales
    )
    head.beta = ridge.fit(design, y, penalty)[0].cpu().numpy()
    head.col_rms = rms[0, 0].cpu().numpy()
    return head

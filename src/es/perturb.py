"""Sampling and aggregation of EGGROLL low-rank perturbations.

Perturbations are drawn in **antithetic pairs**: member ``i`` gets ``+E`` and
member ``i + N/2`` gets ``-E`` (obtained by negating ``A``, since
``E = A B^T / sqrt(r)``). Mirroring cancels the odd part of the fitness
landscape and is the cheapest variance reduction available -- it matters far
more here than in the paper's LLM runs, because a population of a few hundred
against a 12M-parameter suffix is a much noisier estimator than theirs.

Factors are kept in memory for the whole population rather than regenerated
from a counter-based RNG at update time. The paper regenerates because its
models are billion-scale; here the entire population's factors for four
perturbed blocks are ~134 MB at N=256, r=4, so storing them is simpler and
removes a whole class of seed-management bugs.
"""

from __future__ import annotations

import torch

from src.es.forward_unimol import OUTPUT_SIDE, ESSpec, Perturbation, SplitUniMol


def _region_mask(out_dim: int, n_regions: int, rank: int, n: int,
                 generator: torch.Generator, device) -> torch.Tensor:
    """``(n, out_dim, rank)`` mask pinning each A column to one head region.

    A dense rank-1 perturbation moves all 512 output dimensions coherently, but
    the head reads 16 disjoint regions. Confining a column to one region makes
    each member ask a sharper question -- "should *this* region change?" -- and
    cuts the effective search dimension per member from 512xIn to 32xIn, for no
    extra compute.
    """
    width = out_dim // n_regions
    row_region = (torch.arange(out_dim, device=device) // width).view(1, -1, 1)
    pick = torch.randint(n_regions, (n, 1, rank), generator=generator,
                         device=device)
    return (row_region == pick).to(torch.float32)


@torch.no_grad()
def sample(split: SplitUniMol, n_members: int, sigma: float,
           generator: torch.Generator, antithetic: bool = True
           ) -> Perturbation:
    """Draw low-rank factors for one population chunk.

    Args:
        n_members: chunk size; must be even when ``antithetic``.
        sigma: global perturbation scale, before per-matrix relative scaling.

    Returns:
        A ``Perturbation`` whose ``sigma`` entries already fold in the
        ``1/sqrt(r)`` normalisation and the per-matrix relative scale, so the
        forward pass just multiplies by one number.
    """
    spec: ESSpec = split.spec
    r = spec.rank
    if antithetic and n_members % 2:
        raise ValueError(f'antithetic sampling needs an even chunk, got {n_members}')
    half = n_members // 2 if antithetic else n_members

    pert = Perturbation()
    for key in split.targets:
        w = split.base[key]
        out_dim, in_dim = w.shape
        dev, dt = w.device, torch.float32

        a = torch.randn(half, out_dim, r, generator=generator, device=dev, dtype=dt)
        b = torch.randn(half, in_dim, r, generator=generator, device=dev, dtype=dt)
        if spec.region_mask and key[1] in OUTPUT_SIDE and out_dim % spec.n_regions == 0:
            a = a * _region_mask(out_dim, spec.n_regions, r, half, generator, dev)
        if antithetic:
            a = torch.cat([a, -a], dim=0)
            b = torch.cat([b, b], dim=0)

        pert.factors[key] = (a.to(split.dtype), b.to(split.dtype))
        pert.sigma[key] = sigma * split.sigma_scale[key] / (r ** 0.5)
    return pert


@torch.no_grad()
def accumulate(pert: Perturbation, fitness: torch.Tensor,
               into: dict[tuple[int, str], torch.Tensor]) -> None:
    """Add this chunk's ``sum_i f_i E_i`` into the running gradient estimate.

    ``sum_i f_i A_i B_i^T`` is computed as ``r`` plain GEMMs of
    ``(out, N) x (N, in)`` rather than a single einsum -- the paper's own
    ``(diag(f) A)^T B`` trick, generalised past rank 1 -- so the aggregation
    never materialises a per-member ``E_i``.
    """
    for key, (a, b) in pert.factors.items():
        af = a.float() * fitness.view(-1, 1, 1)
        acc = into.get(key)
        for j in range(a.shape[-1]):
            term = af[:, :, j].t() @ b[:, :, j].float()
            acc = term if acc is None else acc + term
        into[key] = acc / (a.shape[-1] ** 0.5)


class RelativeAdam:
    """Adam over ES gradients, with a per-matrix *relative* step size.

    Adam normalises away the gradient scale, so a single learning rate would
    move every matrix by the same absolute amount regardless of how large its
    pretrained weights are. Scaling the step by ``||W||_F / sqrt(mn)`` -- the
    same factor used for sigma -- makes the learning rate mean "fraction of
    typical weight magnitude per step", which is the only setting that
    transfers across the 512x512 and 512x2048 matrices in one model.
    """

    def __init__(self, keys, shapes, sigma_scale, device, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr, self.betas, self.eps = lr, betas, eps
        self.sigma_scale = sigma_scale
        self.t = 0
        self.m = {k: torch.zeros(shapes[k], device=device) for k in keys}
        self.v = {k: torch.zeros(shapes[k], device=device) for k in keys}

    @torch.no_grad()
    def step(self, split: SplitUniMol,
             grads: dict[tuple[int, str], torch.Tensor]) -> float:
        """Apply one ascent step. Returns the mean relative update magnitude."""
        b1, b2 = self.betas
        self.t += 1
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        total = 0.0
        for key, g in grads.items():
            m = self.m[key].mul_(b1).add_(g, alpha=1 - b1)
            v = self.v[key].mul_(b2).addcmul_(g, g, value=1 - b2)
            direction = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            step = self.lr * self.sigma_scale[key] * direction
            split.mean(key).add_(step.to(split.mean(key).dtype))
            total += float(step.abs().mean()) / max(self.sigma_scale[key], 1e-12)
        return total / max(len(grads), 1)

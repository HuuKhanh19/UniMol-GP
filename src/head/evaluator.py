"""Lock-step batched evaluation of a postfix tree population.

The whole population executes together, one token *position* at a time. At
position ``t`` every tree does whatever its own opcode says, selected with
``torch.where`` over the handful of opcodes that actually occur at that
position, so there is no Python-level per-tree loop and no kernel launch that
scales with ``P``.

Cost is ``O(max_len)`` kernel launches on ``(P, n)`` tensors, independent of
population size. For P=400, n=900 that is a few milliseconds on a GPU -- the
reason this project does not need a CUDA genetic-programming library.
"""

from __future__ import annotations

import numpy as np
import torch

from src.head import ops as O
from src.head.genome import Population, stack_depths

#: Tree outputs whose standard deviation falls below this are constants in
#: disguise; they are collinear with the ridge intercept and carry no signal.
DEGENERATE_STD = 1e-8


def _positions_in_use(code: np.ndarray) -> list[np.ndarray]:
    """Opcodes actually present at each token position, for dispatch pruning."""
    return [np.unique(code[:, t]) for t in range(code.shape[1])]


def evaluate(pop: Population, x: torch.Tensor) -> torch.Tensor:
    """Evaluate every tree in ``pop`` on every row of ``x``.

    Args:
        pop: population whose ``arg`` entries are *global* column indices into
            ``x`` (region restriction is enforced when genomes are built).
        x: ``(n, d)`` feature matrix on the target device.

    Returns:
        ``(P, n)`` tensor of tree outputs, finite everywhere.
    """
    device = x.device
    n_rows = x.shape[0]
    p = pop.size
    t_max = int(pop.length.max())
    if t_max == 0:
        return torch.zeros(p, n_rows, device=device)

    code_np = pop.code[:, :t_max]
    sp = stack_depths(code_np)
    # Write slot: sp - arity for real opcodes (terminals push at sp, unary
    # overwrite their operand, binary overwrite their left operand). NOP is
    # pinned to slot 0 and copies it back, which is a no-op.
    w_np = np.where(code_np == O.NOP, 0, sp - O.ARITY[code_np])
    max_stack = int(max(sp.max() + 1, w_np.max() + 2, 2))
    w_np = np.clip(w_np, 0, max_stack - 2)

    code = torch.from_numpy(code_np).to(device)
    arg = torch.from_numpy(pop.arg[:, :t_max]).to(device)
    const = torch.from_numpy(pop.const[:, :t_max]).to(device=device, dtype=x.dtype)
    w = torch.from_numpy(w_np).to(device)

    xt = x.t().contiguous()  # (d, n) so a VAR gather is one index_select
    stack = torch.zeros(p, max_stack, n_rows, device=device, dtype=x.dtype)
    present = _positions_in_use(code_np)

    for t in range(t_max):
        here = present[t]
        if here.size == 1 and here[0] == O.NOP:
            continue

        code_t = code[:, t].unsqueeze(1)  # (P, 1) broadcasts over rows
        w_t = w[:, t]
        idx_a = w_t.view(p, 1, 1).expand(p, 1, n_rows)
        a = stack.gather(1, idx_a).squeeze(1)

        res = a  # NOP / default: leave the slot untouched
        if O.VAR in here:
            res = torch.where(code_t == O.VAR, xt.index_select(0, arg[:, t]), res)
        if O.CONST in here:
            res = torch.where(code_t == O.CONST, const[:, t].unsqueeze(1), res)

        binary = [op for op in here if O.ARITY[op] == 2]
        if binary:
            idx_b = (w_t + 1).view(p, 1, 1).expand(p, 1, n_rows)
            b = stack.gather(1, idx_b).squeeze(1)
            for op in binary:
                res = torch.where(code_t == op, O.apply_binary(int(op), a, b), res)
        for op in (o for o in here if O.ARITY[o] == 1):
            res = torch.where(code_t == op, O.apply_unary(int(op), a), res)

        res = torch.nan_to_num(res, nan=0.0, posinf=O.OUT_CLAMP, neginf=-O.OUT_CLAMP)
        res = res.clamp_(-O.OUT_CLAMP, O.OUT_CLAMP)
        stack.scatter_(1, idx_a, res.unsqueeze(1))

    return stack[:, 0, :]


def normalise(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescale each tree's output to unit RMS and flag degenerate trees.

    Without this a tree emitting values around 1e3 is effectively unregularised
    relative to one emitting 1e-3, so ridge would rank trees by output scale
    rather than by predictive content. Only the scale is shared across rows --
    no target information is involved -- and the intercept column absorbs any
    offset, so no centring is needed here.

    Returns:
        ``(phi_scaled, degenerate)`` where ``degenerate`` is ``(P,)`` bool for
        trees that are constant and therefore collinear with the intercept.
    """
    rms = phi.pow(2).mean(dim=1, keepdim=True).sqrt()
    degenerate = phi.std(dim=1) < DEGENERATE_STD
    return phi / rms.clamp_min(1e-12), degenerate

"""Operator set for the symbolic (GP) head.

A tree is stored as a *postfix* (reverse-Polish) token sequence, so a whole
population is three dense arrays of shape ``(P, max_len)`` and can be evaluated
by one lock-step pass over token positions -- see ``evaluator.py``.

Every operator is *protected*: it must return a finite value for any finite
input, because a single NaN would poison the ridge solve that scores the whole
population. Guarding here (rather than filtering afterwards) keeps the
evaluator branch-free.
"""

from __future__ import annotations

import numpy as np

# --- opcodes -----------------------------------------------------------------
# Terminals and NOP have arity 0; the evaluator derives stack motion from arity
# alone, so adding an operator means appending to OPS and nothing else.

NOP = 0
VAR = 1
CONST = 2
ADD = 3
SUB = 4
MUL = 5
DIV = 6
TANH = 7
EXP = 8
LOG = 9
SQRT = 10
SQUARE = 11

#: opcode -> (name, arity). Order defines the integer encoding; do not reorder.
OPS: tuple[tuple[str, int], ...] = (
    ('nop', 0),
    ('var', 0),
    ('const', 0),
    ('add', 2),
    ('sub', 2),
    ('mul', 2),
    ('div', 2),
    ('tanh', 1),
    ('exp', 1),
    ('log', 1),
    ('sqrt', 1),
    ('square', 1),
)

N_OPS = len(OPS)
OP_NAMES = tuple(name for name, _ in OPS)
ARITY = np.array([arity for _, arity in OPS], dtype=np.int64)

#: How many values each opcode leaves on the stack. Everything pushes its
#: result except NOP, which is tail padding and must not move the stack -- a
#: pushing NOP would overflow max_stack on short trees.
PUSHES = np.ones(N_OPS, dtype=np.int64)
PUSHES[NOP] = 0

#: Net stack motion of each opcode. ``sp_after = sp_before + DELTA[op]``.
DELTA = PUSHES - ARITY

#: Opcodes a mutation may pick as an internal node, grouped by arity.
BINARY_OPS = tuple(op for op in range(N_OPS) if ARITY[op] == 2)
UNARY_OPS = tuple(op for op in range(N_OPS) if ARITY[op] == 1)
INTERNAL_OPS = BINARY_OPS + UNARY_OPS
TERMINAL_OPS = (VAR, CONST)

#: Infix rendering for ``genome.to_infix``.
_INFIX = {ADD: '+', SUB: '-', MUL: '*', DIV: '/'}

# --- numerical guards --------------------------------------------------------
#: Denominators below this magnitude are treated as this magnitude (signed).
DIV_EPS = 1e-6
#: ``exp`` argument clamp. exp(20) ~ 4.9e8, comfortably inside fp32 range even
#: after a few more multiplications upstream.
EXP_CLAMP = 20.0
#: ``log`` shift, so log_abs(0) = log(LOG_EPS) is finite rather than -inf.
LOG_EPS = 1e-6
#: Every operator output is clamped to this magnitude. Without it a chain of
#: ``square`` nodes overflows fp32 and the ridge Gram matrix goes non-finite.
OUT_CLAMP = 1e6


def apply_unary(op: int, x):
    """Apply unary opcode ``op`` elementwise to a torch tensor."""
    import torch

    if op == TANH:
        return torch.tanh(x)
    if op == EXP:
        return torch.exp(torch.clamp(x, -EXP_CLAMP, EXP_CLAMP))
    if op == LOG:
        return torch.log(torch.abs(x) + LOG_EPS)
    if op == SQRT:
        return torch.sqrt(torch.abs(x))
    if op == SQUARE:
        return x * x
    raise ValueError(f'not a unary opcode: {op}')


def apply_binary(op: int, a, b):
    """Apply binary opcode ``op`` elementwise to two torch tensors."""
    import torch

    if op == ADD:
        return a + b
    if op == SUB:
        return a - b
    if op == MUL:
        return a * b
    if op == DIV:
        # Protected division: push |b| out to DIV_EPS while keeping its sign,
        # so the result stays continuous in b except at the sign flip.
        safe = torch.where(
            torch.abs(b) < DIV_EPS,
            torch.where(b < 0, -DIV_EPS, DIV_EPS).to(b.dtype),
            b,
        )
        return a / safe
    raise ValueError(f'not a binary opcode: {op}')


def infix_symbol(op: int) -> str | None:
    """Infix symbol for ``op``, or None if it renders as a function call."""
    return _INFIX.get(op)

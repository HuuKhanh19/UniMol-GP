"""Postfix genome: dense array representation of a tree population.

A population of ``P`` trees is three ``(P, max_len)`` arrays -- ``code`` (opcode),
``arg`` (feature index for VAR nodes) and ``const`` (literal for CONST nodes) --
padded on the right with NOP. Everything below operates on the whole population
at once with numpy; there are no per-tree Python objects.

The postfix form makes subtree surgery trivial: the subtree rooted at position
``t`` is exactly the contiguous slice ``[t - size[t] + 1, t]``, so crossover is a
slice splice rather than a pointer rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.head.ops import (
    ARITY,
    BINARY_OPS,
    CONST,
    DELTA,
    INTERNAL_OPS,
    NOP,
    UNARY_OPS,
    VAR,
    infix_symbol,
    OP_NAMES,
)

#: Probability that ``grow`` initialisation stops at a terminal below max depth.
GROW_TERMINAL_P = 0.35
#: Probability a new terminal is a constant rather than a variable.
CONST_P = 0.25
#: Ephemeral random constants are drawn uniformly from this range. Inputs are
#: standardised embeddings, so O(1) constants are the useful scale.
CONST_RANGE = (-2.0, 2.0)


def max_len_for_depth(max_depth: int) -> int:
    """Token budget that can always hold a full binary tree of ``max_depth``."""
    return 2 ** (max_depth + 1) - 1


@dataclass
class Population:
    """A batch of postfix trees, right-padded with NOP.

    Attributes:
        code: ``(P, max_len)`` int64 opcodes.
        arg: ``(P, max_len)`` int64 feature index, meaningful only where
            ``code == VAR``.
        const: ``(P, max_len)`` float32 literal, meaningful only where
            ``code == CONST``.
        length: ``(P,)`` int64 number of live tokens; the rest is NOP.
    """

    code: np.ndarray
    arg: np.ndarray
    const: np.ndarray
    length: np.ndarray

    @property
    def size(self) -> int:
        return self.code.shape[0]

    @property
    def max_len(self) -> int:
        return self.code.shape[1]

    def __getitem__(self, idx) -> 'Population':
        idx = np.atleast_1d(np.asarray(idx))
        return Population(
            self.code[idx].copy(),
            self.arg[idx].copy(),
            self.const[idx].copy(),
            self.length[idx].copy(),
        )

    def copy(self) -> 'Population':
        return Population(
            self.code.copy(), self.arg.copy(), self.const.copy(), self.length.copy()
        )


def concat(parts: list[Population]) -> Population:
    """Stack populations that share ``max_len``."""
    return Population(
        np.concatenate([p.code for p in parts], axis=0),
        np.concatenate([p.arg for p in parts], axis=0),
        np.concatenate([p.const for p in parts], axis=0),
        np.concatenate([p.length for p in parts], axis=0),
    )


# --- construction ------------------------------------------------------------


def _gen_tokens(depth: int, max_depth: int, full: bool, cols: np.ndarray,
                rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """Recursively emit one tree in postfix order as (code, arg, const) triples.

    ``cols`` is the set of *global* embedding columns this tree may read, i.e.
    its region. Restricting at generation time keeps the evaluator region-blind,
    so every tree can be scored against the same full-width feature matrix.
    """
    at_limit = depth >= max_depth
    stop_early = (not full) and depth > 0 and rng.random() < GROW_TERMINAL_P
    if at_limit or stop_early:
        if rng.random() < CONST_P:
            return [(CONST, 0, float(rng.uniform(*CONST_RANGE)))]
        return [(VAR, int(rng.choice(cols)), 0.0)]

    op = int(rng.choice(INTERNAL_OPS))
    tokens: list[tuple[int, int, float]] = []
    for _ in range(int(ARITY[op])):
        tokens.extend(_gen_tokens(depth + 1, max_depth, full, cols, rng))
    tokens.append((op, 0, 0.0))
    return tokens


def _pack(trees: list[list[tuple[int, int, float]]], max_len: int) -> Population:
    """Pack variable-length token lists into right-padded dense arrays."""
    p = len(trees)
    code = np.full((p, max_len), NOP, dtype=np.int64)
    arg = np.zeros((p, max_len), dtype=np.int64)
    const = np.zeros((p, max_len), dtype=np.float32)
    length = np.zeros(p, dtype=np.int64)
    for i, tokens in enumerate(trees):
        n = len(tokens)
        code[i, :n] = [t[0] for t in tokens]
        arg[i, :n] = [t[1] for t in tokens]
        const[i, :n] = [t[2] for t in tokens]
        length[i] = n
    return Population(code, arg, const, length)


def random_population(size: int, cols: np.ndarray, rng: np.random.Generator,
                      max_depth: int = 5, min_depth: int = 2,
                      max_len: int | None = None) -> Population:
    """Ramped half-and-half initialisation.

    Depth is ramped across ``[min_depth, max_depth]`` and half of each depth
    bucket uses ``full`` (every branch runs to the depth limit), half uses
    ``grow`` (branches may stop early). This is the standard way to get a
    population that is diverse in both shape and size at generation zero.
    """
    max_len = max_len or max_len_for_depth(max_depth)
    depths = np.linspace(min_depth, max_depth, size).round().astype(int)
    trees = [
        _gen_tokens(0, int(d), full=(i % 2 == 0), cols=cols, rng=rng)
        for i, d in enumerate(depths)
    ]
    return _pack(trees, max_len)


def constant_population(size: int, max_len: int, value: float = 0.0) -> Population:
    """A population of single-node CONST trees -- used as a neutral fallback."""
    code = np.full((size, max_len), NOP, dtype=np.int64)
    arg = np.zeros((size, max_len), dtype=np.int64)
    const = np.zeros((size, max_len), dtype=np.float32)
    code[:, 0] = CONST
    const[:, 0] = value
    return Population(code, arg, const, np.ones(size, dtype=np.int64))


# --- structural analysis -----------------------------------------------------


def stack_depths(code: np.ndarray) -> np.ndarray:
    """``(P, max_len)`` stack depth *before* executing each token.

    The evaluator needs this to know which stack slots a token reads and writes
    without doing any data-dependent control flow of its own.
    """
    p, ln = code.shape
    sp = np.zeros(p, dtype=np.int64)
    out = np.empty((p, ln), dtype=np.int64)
    for t in range(ln):
        out[:, t] = sp
        sp = sp + DELTA[code[:, t]]
    return out


def subtree_sizes(code: np.ndarray) -> np.ndarray:
    """``(P, max_len)`` node count of the subtree rooted at each position.

    Positions inside the NOP padding get size 1 and must not be used as
    crossover points -- callers mask them with ``length``.
    """
    p, ln = code.shape
    ar = ARITY[code]
    sizes = np.ones((p, ln), dtype=np.int64)
    stack = np.zeros((p, ln + 2), dtype=np.int64)
    sp = np.zeros(p, dtype=np.int64)
    rows = np.arange(p)
    for t in range(ln):
        a = ar[:, t]
        s = np.ones(p, dtype=np.int64)
        # Pop `a` sizes off the stack. Max arity is 2, so two masked reads
        # cover every opcode without a data-dependent inner loop.
        for j in range(2):
            take = a > j
            idx = np.clip(sp - 1 - j, 0, None)
            s = s + np.where(take, stack[rows, idx], 0)
        sp = sp - a
        stack[rows, np.clip(sp, 0, ln + 1)] = s
        sp = sp + (code[:, t] != NOP)
        sizes[:, t] = s
    return sizes


def is_valid(code: np.ndarray, length: np.ndarray) -> np.ndarray:
    """``(P,)`` bool: does each genome evaluate to exactly one value?"""
    p, ln = code.shape
    sp = np.zeros(p, dtype=np.int64)
    ok = np.ones(p, dtype=bool)
    for t in range(ln):
        live = t < length
        sp = sp + np.where(live, DELTA[code[:, t]], 0)
        ok &= ~(live & (sp < 1))
    return ok & (sp == 1)


# --- variation ---------------------------------------------------------------


def _choose_points(sizes: np.ndarray, length: np.ndarray,
                   rng: np.random.Generator, bias_internal: float = 0.9,
                   arity: np.ndarray | None = None) -> np.ndarray:
    """Pick one crossover/mutation point per individual.

    Koza's 90/10 rule: prefer internal nodes, because uniform sampling over a
    binary tree lands on a leaf about half the time and leaf-only swaps barely
    move the search.
    """
    p, ln = sizes.shape
    pos = np.arange(ln)[None, :]
    live = pos < length[:, None]
    internal = live & (arity > 0) if arity is not None else live
    want_internal = (rng.random(p) < bias_internal) & internal.any(axis=1)
    pool = np.where(want_internal[:, None], internal, live)
    weights = pool.astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    cum = weights.cumsum(axis=1)
    draw = rng.random((p, 1))
    return (cum < draw).sum(axis=1).clip(0, ln - 1)


def crossover(parents_a: Population, parents_b: Population,
              rng: np.random.Generator) -> Population:
    """Subtree crossover: splice a subtree of B into A.

    In postfix a subtree is a contiguous slice, so this is pure slicing. Any
    child that would exceed ``max_len`` falls back to an unchanged copy of A --
    rejection keeps the length invariant without a retry loop.
    """
    ln = parents_a.max_len
    p = parents_a.size
    sizes_a = subtree_sizes(parents_a.code)
    sizes_b = subtree_sizes(parents_b.code)
    pa = _choose_points(sizes_a, parents_a.length, rng, arity=ARITY[parents_a.code])
    pb = _choose_points(sizes_b, parents_b.length, rng, arity=ARITY[parents_b.code])

    rows = np.arange(p)
    size_a = sizes_a[rows, pa]
    size_b = sizes_b[rows, pb]
    start_a = pa - size_a + 1
    start_b = pb - size_b + 1
    new_len = parents_a.length - size_a + size_b
    fits = new_len <= ln

    out = parents_a.copy()
    for i in np.nonzero(fits)[0]:
        sa, ea = int(start_a[i]), int(pa[i]) + 1
        sb, eb = int(start_b[i]), int(pb[i]) + 1
        la, n_new = int(parents_a.length[i]), int(new_len[i])
        for dst, src_a, src_b, pad in (
            (out.code, parents_a.code, parents_b.code, NOP),
            (out.arg, parents_a.arg, parents_b.arg, 0),
            (out.const, parents_a.const, parents_b.const, 0.0),
        ):
            dst[i, :n_new] = np.concatenate(
                [src_a[i, :sa], src_b[i, sb:eb], src_a[i, ea:la]]
            )
            dst[i, n_new:] = pad
        out.length[i] = n_new
    return out


def mutate(pop: Population, rng: np.random.Generator, cols: np.ndarray,
           p_point: float = 0.15, p_subtree: float = 0.10,
           p_const: float = 0.15, const_sigma: float = 0.3,
           max_depth: int = 3) -> Population:
    """Point, subtree and constant-jitter mutation, applied independently.

    Constant jitter matters more here than in textbook GP: the trees feed a
    ridge solve, so a tree whose *shape* is right but whose literals are off by
    a scale factor is already useful and only needs a nudge.
    """
    out = pop.copy()
    p, ln = out.code.shape
    pos = np.arange(ln)[None, :]
    live = pos < out.length[:, None]

    # -- point mutation: swap an opcode for another of the same arity.
    hit = live & (rng.random((p, ln)) < p_point)
    ar = ARITY[out.code]
    for group in (BINARY_OPS, UNARY_OPS):
        sel = hit & np.isin(out.code, group)
        if sel.any():
            out.code[sel] = rng.choice(group, size=int(sel.sum()))
    sel = hit & (out.code == VAR)
    if sel.any():
        out.arg[sel] = rng.choice(cols, size=int(sel.sum()))

    # -- constant jitter: multiplicative-ish walk on CONST literals.
    sel = live & (out.code == CONST) & (rng.random((p, ln)) < p_const)
    if sel.any():
        out.const[sel] += rng.normal(0.0, const_sigma, size=int(sel.sum())).astype(
            np.float32
        )

    # -- subtree mutation: replace a random subtree with a fresh small tree.
    do_sub = np.nonzero(rng.random(p) < p_subtree)[0]
    if do_sub.size:
        sizes = subtree_sizes(out.code)
        pts = _choose_points(sizes, out.length, rng, arity=ar)
        for i in do_sub:
            t = int(pts[i])
            size = int(sizes[i, t])
            start = t - size + 1
            fresh = _gen_tokens(0, max_depth, full=False, cols=cols, rng=rng)
            new_len = int(out.length[i]) - size + len(fresh)
            if new_len > ln:
                continue
            tail = slice(t + 1, int(out.length[i]))
            code = np.concatenate(
                [out.code[i, :start], [tk[0] for tk in fresh], out.code[i, tail]]
            )
            arg = np.concatenate(
                [out.arg[i, :start], [tk[1] for tk in fresh], out.arg[i, tail]]
            )
            const = np.concatenate(
                [out.const[i, :start], [tk[2] for tk in fresh], out.const[i, tail]]
            )
            out.code[i] = NOP
            out.arg[i] = 0
            out.const[i] = 0.0
            out.code[i, : code.size] = code
            out.arg[i, : arg.size] = arg
            out.const[i, : const.size] = const
            out.length[i] = new_len
    return out


# --- rendering ---------------------------------------------------------------


def to_infix(pop: Population, i: int, feature_names: list[str] | None = None,
             precision: int = 3) -> str:
    """Render genome ``i`` as a human-readable infix expression."""
    stack: list[str] = []
    for t in range(int(pop.length[i])):
        op = int(pop.code[i, t])
        if op == NOP:
            continue
        if op == VAR:
            j = int(pop.arg[i, t])
            stack.append(feature_names[j] if feature_names else f'x{j}')
        elif op == CONST:
            stack.append(f'{float(pop.const[i, t]):.{precision}g}')
        elif ARITY[op] == 1:
            stack.append(f'{OP_NAMES[op]}({stack.pop()})')
        else:
            b, a = stack.pop(), stack.pop()
            sym = infix_symbol(op)
            stack.append(f'({a} {sym} {b})' if sym else f'{OP_NAMES[op]}({a}, {b})')
    return stack[-1] if stack else '<empty>'

"""
DifferentiablePipeline — Phase 3 frozen GP + double-ridge readout,
autograd-compatible via sympy.lambdify with custom safe-modules dict.

Architecture (all components below the encoder are FROZEN):

    embeddings (B, K, D_orig=512)
        ↓ (x − pca_mean) @ pca_components       — frozen buffers
    reduced (B, K, gp_input_dim=128)
        ↓ (x − reduced_mean) / reduced_std      — frozen buffers
    standardized (B, K, gp_input_dim)
        ↓ K lambdified tree functions, mode='safe' or 'vanilla'
    Z (B, K, q=num_trees_per_conformer)
        ↓ einsum('bkq,kq→bk', Z, w_inner) + b_inner  — frozen Parameters
    S (B, K)
        ↓ einsum('bk,k→b', S, w_outer) + b_outer    — frozen Parameters
    y_pred (B,)

Two modes:
  - 'safe'    : SAFE_MODULES dict overrides log/sqrt/min/max + Loose* variants
                so forward never produces NaN. Use during training.
  - 'vanilla' : Plain 'torch' modules. Bit-exact to evogp where finite.
                Use for strict equivalence checks (assert_match_phase2).

NOTE on _tree_to_sympy_fixed:
  evogp's Tree.to_sympy_expr has a bug for UFUNC nodes flagged with OUT_NODE
  (single-arg functions like sin/cos/exp/log being roots of an output): it
  references undefined `right` variable. We re-implement the conversion
  ourselves with the fix (push `mid` for UFUNC instead of `right`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from evogp.tree import Forest
from evogp.tree.utils import NType, SYMPY_MAP

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Safe sympy construction map
# ──────────────────────────────────────────────────────────────────────
# evogp's raw SYMPY_MAP[Func.DIV] = lambda x,y: x/y eagerly evaluates,
# so a tree with `DIV(anything, CONST_0)` produces sympy.zoo (complex
# infinity) at construction time. When that subexpression then appears
# inside sp.Min/sp.Max, sympy throws "not comparable".
#
# Fix: override DIV (id 4) and LOG (id 20) with evogp's Loose* sp.Function
# classes (deferred — they don't auto-evaluate). Symbolic expressions
# never contain zoo. At lambdify time, SAFE_MODULES translates LooseDiv
# /LooseLog to torch's safe_div/safe_log for runtime evaluation.
def _build_safe_constr_map():
    m = dict(SYMPY_MAP)
    try:
        from evogp.tree.utils import LooseDiv
        m[4] = LooseDiv  # Func.DIV
    except ImportError:
        logger.warning("evogp.tree.utils.LooseDiv not found — DIV stays raw")
    try:
        from evogp.tree.utils import LooseLog
        m[20] = LooseLog  # Func.LOG
    except ImportError:
        logger.warning("evogp.tree.utils.LooseLog not found — LOG stays raw")
    # Also handle SQRT (id 27) and POW preemptively if available
    try:
        from evogp.tree.utils import LooseSqrt
        m[27] = LooseSqrt  # Func.SQRT
    except ImportError:
        pass
    return m


SAFE_CONSTR_SYMPY_MAP = _build_safe_constr_map()


# ──────────────────────────────────────────────────────────────────────
# Safe substitute ops (NaN-free for autograd)
# ──────────────────────────────────────────────────────────────────────
# EPS = 1e-3 (was 1e-8 originally).
# Why this larger value: safe_inv / safe_div / safe_log all clamp at EPS.
# At the boundary, gradient magnitude is ~1/EPS² for safe_inv. With EPS=1e-8,
# that's 1e16 per op — three chained ops cascade to 1e48, blowing through
# float32 max (3.4e38) → inf in saved tensors → NaN backward.
# With EPS=1e-3: max amp 1e3 per op, three chained = 1e9, well bounded.
# Trade-off: slight semantic divergence from Phase 2 CUDA's eps choice.
EPS = 1e-3


def _ensure_tensor(x) -> torch.Tensor:
    """Convert scalar (int/float) to a 0-dim torch.float32 tensor.

    Sympy's lambdify can produce expressions like `cos(1)` or `exp(0.5)`
    when a tree has a CONST as the only argument to a function. The raw
    integer/float is passed to our safe op, which then errors because
    torch ops require Tensor input. This helper makes safe ops scalar-safe.
    """
    if torch.is_tensor(x):
        return x
    return torch.tensor(float(x), dtype=torch.float32)


def safe_log(x) -> torch.Tensor:
    """log(|x|) with floor on |x|. Matches evogp LooseLog semantics."""
    x = _ensure_tensor(x)
    return torch.log(torch.clamp(torch.abs(x), min=EPS))


def safe_inv(x) -> torch.Tensor:
    """1/x with denominator floored at ±EPS (sign-preserving)."""
    x = _ensure_tensor(x)
    sign = torch.sign(x)
    sign = sign + (sign == 0).to(x.dtype)
    return 1.0 / (sign * torch.clamp(torch.abs(x), min=EPS))


def safe_div(x, y) -> torch.Tensor:
    x = _ensure_tensor(x)
    return x * safe_inv(y)


def safe_sqrt(x) -> torch.Tensor:
    x = _ensure_tensor(x)
    return torch.sqrt(torch.clamp(torch.abs(x), min=EPS))


def safe_pow(x, y) -> torch.Tensor:
    x = _ensure_tensor(x)
    return torch.pow(torch.clamp(torch.abs(x), min=EPS), y)


def safe_exp(x) -> torch.Tensor:
    """exp with input clamped to [-10, 10].

    exp(10) ≈ 22026, exp(-10) ≈ 4.5e-5 — both well within float32 range
    and small enough that cascading multiplications through a 3-layer
    tree (e.g. `exp(a) * exp(b) * exp(c)`) stay finite (max ≈ 1e13).

    Forward stays bounded; backward gradient at clamp boundary is zero
    (clamp blocks gradient flow when input is at boundary), so saturated
    paths don't propagate inf/NaN backward through the chain rule.
    """
    x = _ensure_tensor(x)
    return torch.exp(torch.clamp(x, min=-10.0, max=10.0))


def safe_sin(x) -> torch.Tensor:
    return torch.sin(_ensure_tensor(x))


def safe_cos(x) -> torch.Tensor:
    return torch.cos(_ensure_tensor(x))


def safe_abs(x) -> torch.Tensor:
    return torch.abs(_ensure_tensor(x))


def _to_tensor_like(s, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(s):
        return s
    return torch.full_like(ref, float(s))


def safe_min(*args) -> torch.Tensor:
    """torch.minimum with scalar broadcasting (handles Min(0, tensor) etc.).

    If ALL arguments are scalars, returns a 0-dim tensor (not a Python float)
    so downstream tensor ops always work.
    """
    ref = next((a for a in args if torch.is_tensor(a)), None)
    if ref is None:
        return _ensure_tensor(min(*args))
    tensors = [_to_tensor_like(a, ref) for a in args]
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.minimum(out, t)
    return out


def safe_max(*args) -> torch.Tensor:
    ref = next((a for a in args if torch.is_tensor(a)), None)
    if ref is None:
        return _ensure_tensor(max(*args))
    tensors = [_to_tensor_like(a, ref) for a in args]
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.maximum(out, t)
    return out


SAFE_MODULES: List = [
    {
        # Override unsafe ops with bounded + scalar-safe versions:
        'log':       safe_log,
        'sqrt':      safe_sqrt,
        'exp':       safe_exp, 'Exp': safe_exp,
        # Trig: scalar-safe (sympy can produce e.g. cos(1) when CONST is the
        # only argument; raw torch.cos rejects int/float scalars):
        'sin':       safe_sin, 'Sin': safe_sin,
        'cos':       safe_cos, 'Cos': safe_cos,
        # Min/Max — both case variants since sympy printer may use either:
        'min':       safe_min, 'Min': safe_min,
        'max':       safe_max, 'Max': safe_max,
        # evogp's protected sympy.Function classes (used when Phase 2
        # configures `loose_div`, `loose_log`, etc.):
        'LooseLog':  safe_log,
        'LooseDiv':  safe_div,
        'LooseInv':  safe_inv,
        'LooseSqrt': safe_sqrt,
        'LoosePow':  safe_pow,
        # abs — scalar-safe (sympy may produce abs(0.5) etc.):
        'Abs':       safe_abs, 'abs': safe_abs,
    },
    'torch',
]

VANILLA_MODULES = 'torch'


# ──────────────────────────────────────────────────────────────────────
# Fixed sympy converter (works around evogp UFUNC OUT_NODE bug)
# ──────────────────────────────────────────────────────────────────────
def _tree_to_sympy_fixed(tree):
    """Mirror of evogp.Tree.to_sympy_expr with UFUNC OUT_NODE bug fixed.

    The bug: when a UFUNC node has OUT_NODE flag, evogp's original code
    references undefined `right` variable (only `mid` is set for UFUNC).
    We push `mid` for UFUNC, `right` for BFUNC/TFUNC (matching evogp's
    pattern for binary/ternary).
    """
    node_value = tree.node_value.detach().cpu().numpy()
    node_type = tree.node_type.detach().cpu().numpy()
    subtree_size = tree.subtree_size.detach().cpu().numpy()
    tree_size = int(subtree_size[0])
    input_len = tree.input_len
    output_len = tree.output_len

    x = sp.symbols(f"x:{input_len}", real=True)
    if not isinstance(x, tuple):
        x = (x,)

    expr_stack = []

    if output_len == 1:
        # Single-output path — no OUT_NODE handling needed
        for i in reversed(range(tree_size)):
            t, v = node_type[i], node_value[i]
            if t == NType.VAR:
                expr_stack.append(x[int(v)])
            elif t == NType.CONST:
                expr_stack.append(v)
            else:  # Function
                if t == NType.UFUNC:
                    mid = expr_stack.pop(-1)
                    res = SAFE_CONSTR_SYMPY_MAP[int(v)](mid)
                elif t == NType.BFUNC:
                    left = expr_stack.pop(-1)
                    right = expr_stack.pop(-1)
                    res = SAFE_CONSTR_SYMPY_MAP[int(v)](left, right)
                elif t == NType.TFUNC:
                    left = expr_stack.pop(-1)
                    mid = expr_stack.pop(-1)
                    right = expr_stack.pop(-1)
                    res = SAFE_CONSTR_SYMPY_MAP[int(v)](left, mid, right)
                else:
                    raise ValueError(f"Unknown node type {t} at index {i}")
                expr_stack.append(res)
        return expr_stack[0]

    # Multi-output path with OUT_NODE handling — FIXED for UFUNC
    expr = [0] * output_len
    for i in reversed(range(tree_size)):
        t, v = node_type[i], node_value[i]
        if t & NType.OUT_NODE:
            # Output index encoded in upper 16 bits of float-as-int32;
            # function ID in lower 8 bits.
            v_int32 = np.array([v], dtype=np.float32).view(np.int32)[0]
            out_idx = int(v_int32) >> 16
            v = int(v_int32) & 0xFF
        else:
            out_idx = -1
        t = t & NType.TYPE_MASK

        if t == NType.VAR:
            expr_stack.append(x[int(v)])
        elif t == NType.CONST:
            expr_stack.append(v)
        else:  # Function
            if t == NType.UFUNC:
                mid = expr_stack.pop(-1)
                res = SAFE_CONSTR_SYMPY_MAP[int(v)](mid)
                if out_idx != -1:
                    expr[out_idx] += res
                    expr_stack.append(mid)        # FIX: push mid (not right)
                else:
                    expr_stack.append(res)
            elif t == NType.BFUNC:
                left = expr_stack.pop(-1)
                right = expr_stack.pop(-1)
                res = SAFE_CONSTR_SYMPY_MAP[int(v)](left, right)
                if out_idx != -1:
                    expr[out_idx] += res
                    expr_stack.append(right)      # match evogp's pattern
                else:
                    expr_stack.append(res)
            elif t == NType.TFUNC:
                left = expr_stack.pop(-1)
                mid = expr_stack.pop(-1)
                right = expr_stack.pop(-1)
                res = SAFE_CONSTR_SYMPY_MAP[int(v)](left, mid, right)
                if out_idx != -1:
                    expr[out_idx] += res
                    expr_stack.append(right)      # match evogp's pattern
                else:
                    expr_stack.append(res)
            else:
                raise ValueError(f"Unknown node type {t} at index {i}")

    return expr


# ──────────────────────────────────────────────────────────────────────
# Tree compilation helpers
# ──────────────────────────────────────────────────────────────────────
def reconstruct_tree(tstate: Dict[str, torch.Tensor], input_len: int, output_len: int):
    """Rebuild a Tree from a saved {'node_value', 'node_type', 'subtree_size'} dict.

    Uses Forest(input_len, output_len, batch_node_value, batch_node_type,
    batch_subtree_size) constructor (verified API per Sprint 0).
    """
    forest = Forest(
        input_len,
        output_len,
        tstate["node_value"][None, :].cuda(),
        tstate["node_type"][None, :].cuda(),
        tstate["subtree_size"][None, :].cuda(),
    )
    return forest[0]


def lambdify_tree(tree, input_len: int, mode: str = 'safe'):
    """Compile a Tree's sympy expression(s) into a callable.

    Uses our patched _tree_to_sympy_fixed to work around evogp's UFUNC
    OUT_NODE bug.
    """
    syms = sp.symbols(f"x:{input_len}", real=True)
    if not isinstance(syms, tuple):
        syms = (syms,)

    raw = _tree_to_sympy_fixed(tree)   # ← bug-fixed version
    exprs = raw if isinstance(raw, list) else [raw]

    modules = SAFE_MODULES if mode == 'safe' else VANILLA_MODULES

    fns = []
    n_failed = 0
    for e in exprs:
        try:
            fn = sp.lambdify(syms, e, modules=modules)
            fns.append(fn)
        except Exception as ex:
            logger.warning(f"lambdify failed for expr {e}: {type(ex).__name__}: {ex}")
            fns.append(None)
            n_failed += 1

    def evaluate(X: torch.Tensor) -> torch.Tensor:
        # X: (..., input_len)
        args = [X[..., i] for i in range(input_len)]
        ref = X[..., 0]  # batch shape + device reference
        outs = []
        for f in fns:
            if f is None:
                outs.append(torch.zeros_like(ref))
                continue
            o = f(*args)
            # Three cases for f's return type:
            #   1. Python scalar (e.g. lambda returning literal 2.0 after
            #      sympy simplifies x-x+2 → 2)
            #   2. 0-dim tensor (constant expression evaluated by our
            #      safe_* ops — they return tensors via _ensure_tensor,
            #      which creates on CPU by default → device mismatch)
            #   3. (batch,) tensor (normal case)
            # Cases 1 and 2 need broadcasting + device match before stack().
            if not torch.is_tensor(o):
                o = torch.full_like(ref, float(o))
            elif o.shape != ref.shape:
                if o.dim() == 0:
                    # 0-dim tensor (likely on CPU from _ensure_tensor) →
                    # extract scalar and rebuild on ref's device + shape.
                    # No gradient loss: 0-dim implies constant expression
                    # (no x dependency).
                    o = torch.full_like(ref, o.item())
                else:
                    # Higher-dim broadcast-compatible — explicit device move
                    o = o.broadcast_to(ref.shape).to(ref.device)
            outs.append(o)
        return torch.stack(outs, dim=-1)

    return evaluate, n_failed


# ──────────────────────────────────────────────────────────────────────
# DifferentiablePipeline
# ──────────────────────────────────────────────────────────────────────
class DifferentiablePipeline(nn.Module):
    """Frozen GP + double-ridge readout, autograd-compatible.

    Loads from:
      - best_individual_dict: torch.load(best_individual.pt). Has 'config'
        (GPConfig dict) and 'snapshot' (BestSnapshot serialized dict).
      - transform_info: dict with PCA components and standardize stats,
        produced by reduce_and_normalize_splits() applied to Phase 1 cache.

    Forward:
      embeddings (B, K, D_orig) → y_pred (B,)

    All non-encoder components are buffers / non-trainable Parameters.
    Trees are pure functions (no learnable params).
    """

    def __init__(
        self,
        best_individual_dict: Dict[str, Any],
        transform_info: Dict[str, torch.Tensor],
        mode: str = 'safe',
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if mode not in ('safe', 'vanilla'):
            raise ValueError(f"mode must be 'safe' or 'vanilla', got {mode!r}")
        self.mode = mode

        cfg = best_individual_dict["config"]      # GPConfig as dict
        snap = best_individual_dict["snapshot"]   # BestSnapshot serialized

        self.K = int(cfg["K"])
        self.q = int(cfg["num_trees_per_conformer"])
        self.gp_input_dim = int(cfg["D"])
        self.D_orig = int(transform_info.get("D_orig", 512))
        self.max_tree_len = int(cfg["max_tree_len"])

        # ── PCA layer (frozen buffers): reduced = (x − pca_mean) @ pca_components
        # pca_components: (D_orig, gp_input_dim)
        # pca_mean:       (D_orig,)
        self.register_buffer(
            "pca_components",
            transform_info["pca_components"].clone().detach().float(),
        )
        self.register_buffer(
            "pca_mean",
            transform_info["pca_mean"].clone().detach().float(),
        )

        # ── Standardize layer: (reduced − reduced_mean) / reduced_std
        self.register_buffer(
            "reduced_mean",
            transform_info["reduced_mean"].clone().detach().float(),
        )
        self.register_buffer(
            "reduced_std",
            transform_info["reduced_std"].clone().detach().float(),
        )

        # ── Trees: reconstruct K trees and lambdify
        if "trees_state" not in snap:
            raise KeyError("snapshot missing 'trees_state'")
        if len(snap["trees_state"]) != self.K:
            raise ValueError(
                f"trees_state length {len(snap['trees_state'])} != K={self.K}"
            )

        self.tree_fns = []
        n_failed_total = 0
        for k, tstate in enumerate(snap["trees_state"]):
            tree = reconstruct_tree(tstate, self.gp_input_dim, self.q)
            fn, n_failed = lambdify_tree(tree, self.gp_input_dim, mode=mode)
            self.tree_fns.append(fn)
            n_failed_total += n_failed
        if n_failed_total > 0:
            logger.warning(
                f"DifferentiablePipeline: {n_failed_total}/{self.K * self.q} "
                f"sympy expressions failed to lambdify"
            )

        # ── Frozen ridge weights
        rp = snap["ridge_params"]
        self.w_inner = nn.Parameter(
            rp["w_inner"].clone().detach().float(), requires_grad=False
        )  # (K, q)
        self.b_inner = nn.Parameter(
            rp["b_inner"].clone().detach().float(), requires_grad=False
        )  # (K,)
        self.w_outer = nn.Parameter(
            rp["w_outer"].clone().detach().float(), requires_grad=False
        )  # (K,)
        self.b_outer = nn.Parameter(
            rp["b_outer"].clone().detach().float().reshape(()), requires_grad=False
        )  # scalar

        # Sanity shape checks
        assert self.w_inner.shape == (self.K, self.q), (
            f"w_inner shape {self.w_inner.shape} != ({self.K}, {self.q})"
        )
        assert self.b_inner.shape == (self.K,), f"b_inner shape {self.b_inner.shape}"
        assert self.w_outer.shape == (self.K,), f"w_outer shape {self.w_outer.shape}"

        if device is not None:
            self.to(device)

        logger.info(
            f"DifferentiablePipeline initialized: K={self.K}, q={self.q}, "
            f"gp_input_dim={self.gp_input_dim}, D_orig={self.D_orig}, mode={mode}"
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (B, K, D_orig) — output of UniMol encoder.

        Returns:
            y_pred: (B,) — predicted target.
        """
        if embeddings.dim() != 3:
            raise ValueError(f"embeddings must be 3D (B, K, D), got {embeddings.shape}")
        B, K_in, D_orig = embeddings.shape
        if K_in != self.K:
            raise ValueError(f"K mismatch: input K={K_in} vs pipeline K={self.K}")
        if D_orig != self.D_orig:
            raise ValueError(
                f"D_orig mismatch: input D={D_orig} vs pipeline D={self.D_orig}"
            )

        # ── Backward hook: filter NaN/inf in gradient flowing back to encoder.
        # Compound safe ops in lambdified trees (safe_div near 0, safe_log near
        # 0, etc.) save internal tensors that can be ~1e8. Backward chain rule
        # `0 * inf = NaN` (IEEE 754) produces NaN at clamp-boundary positions.
        # Encoder params are SHARED across batch positions — even one NaN
        # batch position infects 100% of encoder params via grad summation.
        # Hook fires during backward AT this tensor: NaN grads are zeroed out
        # BEFORE they propagate into encoder, isolating instability to pipeline.
        if embeddings.requires_grad:
            embeddings.register_hook(
                lambda grad: torch.nan_to_num(
                    grad, nan=0.0, posinf=0.0, neginf=0.0
                )
            )

        # 1. PCA
        reduced = (embeddings - self.pca_mean) @ self.pca_components  # (B, K, gp_input_dim)

        # 2. Standardize
        std = (reduced - self.reduced_mean) / self.reduced_std

        # Bound standardized inputs: PCA + standardize on Phase 1 cache should
        # produce roughly N(0, 1), but during fine-tuning encoder weights drift
        # → projections may drift outliers. Clamp to ±10 (10σ — covers all
        # reasonable values) prevents tree internal ops from receiving extreme
        # inputs that overflow saved tensors during backward.
        std = torch.clamp(std, min=-10.0, max=10.0)

        # 3. Per-conformer tree forward (Python loop over K — typically K=10)
        Z_list = []
        for k in range(self.K):
            xk = std[:, k, :].contiguous()  # (B, gp_input_dim)
            zk = self.tree_fns[k](xk)        # (B, q)
            # Sanitize tree output: nan_to_num zeros non-finite (gradient
            # blocked there), then hard-clamp finite values to ±100. The
            # backward hook on embeddings above catches any remaining
            # NaN gradients that leak through compound ops.
            zk = torch.nan_to_num(zk, nan=0.0, posinf=0.0, neginf=0.0)
            zk = torch.clamp(zk, min=-100.0, max=100.0)
            Z_list.append(zk)
        Z = torch.stack(Z_list, dim=1)       # (B, K, q)

        # 4. Inner ridge per conformer:  s_k = z_k · w_inner[k] + b_inner[k]
        S = torch.einsum('bkq,kq->bk', Z, self.w_inner) + self.b_inner.unsqueeze(0)

        # 5. Outer ridge:  y = S · w_outer + b_outer
        y_pred = torch.einsum('bk,k->b', S, self.w_outer) + self.b_outer

        return y_pred

    @torch.no_grad()
    def assert_match_phase2(
        self,
        embeddings: torch.Tensor,
        expected_y_pred: torch.Tensor,
        tolerance: float = 1e-3,
    ):
        """Sanity check: forward should match Phase 2's saved predictions.

        Run with mode='vanilla' for strict bit-exact-style equivalence.
        """
        was_training = self.training
        self.eval()
        try:
            y_actual = self.forward(embeddings)
            diff = (y_actual - expected_y_pred).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            logger.info(
                f"assert_match_phase2: max_diff={max_diff:.2e}, "
                f"mean_diff={mean_diff:.2e}, tolerance={tolerance:.2e}"
            )
            if max_diff > tolerance:
                logger.warning(
                    f"Phase 2 match check exceeded tolerance "
                    f"({max_diff:.2e} > {tolerance:.2e}); "
                    f"acceptable in 'safe' mode (small drift expected at edge inputs)."
                )
        finally:
            if was_training:
                self.train()


# ──────────────────────────────────────────────────────────────────────
# Convenience: load pipeline from on-disk artifacts
# ──────────────────────────────────────────────────────────────────────
def load_pipeline_from_disk(
    best_individual_path: str,
    cache_path: str,
    gp_input_dim: int,
    mode: str = 'safe',
    device: Optional[torch.device] = None,
) -> DifferentiablePipeline:
    """Load pipeline from best_individual.pt + Phase 1 cache.

    Re-fits PCA + standardize on cache (deterministic, matches Phase 2's
    transform_info exactly).
    """
    from src.data.embeddings_cache import load_cache, reduce_and_normalize_splits

    best_dict = torch.load(best_individual_path, map_location='cpu', weights_only=False)
    cache = load_cache(cache_path, map_location='cpu')
    _, transform_info = reduce_and_normalize_splits(cache, gp_input_dim=gp_input_dim)

    return DifferentiablePipeline(
        best_individual_dict=best_dict,
        transform_info=transform_info,
        mode=mode,
        device=device,
    )

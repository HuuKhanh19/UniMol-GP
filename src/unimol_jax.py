"""
UniMol v1 forward pass ported to JAX for EGGROLL fine-tuning (Task 2).

This module re-implements the `molecule` configuration of UniMol v1
(`unimol_source/unimol_tools/models/unimol.py` + `transformers.py`) as a tree
of EGGROLL `Model` classes, so it can be optimised by the EGGROLL Noiser
instead of backprop.

Design (decided in Task 1 + the EGGROLL-interface review):

  * The pretrained encoder weight W of every projection is FROZEN. It is read
    directly from `common_params.params['W']` and applied with a plain
    `x @ W.T`. It is NEVER routed through `noiser.do_mm`, because `do_mm`
    perturbs every matrix it touches regardless of `es_map`.
  * Only the LoRA adapters P, Q go through `do_mm` (es_map MM_PARAM), so only
    they receive EGGROLL's low-rank perturbation / update.
  * The new ESOL head is trained from scratch: its weights are MM_PARAM
    (perturbed/updated by EGGROLL's low-rank mechanism, but as a single full
    matrix -- no frozen base, no P/Q adapter) and its biases are PARAM.
  * `freeze_nonlora` MUST be False (see module note). Encoder biases and all
    LayerNorm parameters are frozen via `es_map = EXCLUDED`, not via the flag,
    so that the head bias (a PARAM) can still train.
  * `embed_tokens` and the Gaussian edge tables are EXCLUDED and indexed
    directly (EggRoll.do_emb is unimplemented).

Parity notes for Task 3 weight-conversion / verification:
  * `gaussian()` reproduces UniMol's truncated constant `pi = 3.14159`.
  * GELU is the exact erf-based variant (`approximate=False`).
  * LayerNorm uses biased variance and eps inside the sqrt (matches torch).
  * The fused torch `in_proj` (512 -> 1536) is split into three independent
    q/k/v projections; that split is performed by Task 3's converter.

The model `_forward` operates on a SINGLE molecule. Batch over the ESOL set
and over the EGGROLL population with `jax.vmap` (Tasks 5-6).
"""

import os
import sys
import importlib

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------
# hyperscalees (EGGROLL) import bootstrap
# --------------------------------------------------------------------------
# This file runs as a top-level module (`python run.py` imports it as
# `unimol_jax`), so relative imports into the hyperscalees package are not
# possible. Importing hyperscalees normally also fails on a CPU-only box,
# because hyperscalees/models/__init__.py does `from . import rl, llm`,
# pulling heavy optional deps (gymnax / distrax).
#
# `_hs_bootstrap` (a sibling file -- keep it next to this one) registers
# lightweight package shims so the leaf modules we actually need import
# cleanly, bypassing those __init__.py files. It does NOT depend on
# `hyperscalees.__file__`, which was the source of the earlier crash.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _hs_bootstrap import bootstrap_hyperscalees

bootstrap_hyperscalees(search_from=_HERE)     # idempotent; run.py may call it first


def _load_hs_module(qualified_name):
    """Import a hyperscalees leaf module, e.g. 'hyperscalees.models.common'.

    Backward-compatible name -- train_eggroll.py imports it. The heavy-__init__
    bypass is handled once by `_hs_bootstrap.bootstrap_hyperscalees`; here we
    just make sure the bootstrap has run, then do a normal import.
    """
    bootstrap_hyperscalees(search_from=_HERE)
    return importlib.import_module(qualified_name)


# Pretrained-encoder building blocks from the EGGROLL package.
_bm = _load_hs_module("hyperscalees.models.base_model")
_cm = _load_hs_module("hyperscalees.models.common")
Model, CommonInit = _bm.Model, _bm.CommonInit
Parameter, MM, Linear = _cm.Parameter, _cm.MM, _cm.Linear
merge_inits, merge_frozen = _cm.merge_inits, _cm.merge_frozen
call_submodule = _cm.call_submodule
PARAM, MM_PARAM = _cm.PARAM, _cm.MM_PARAM
EMB_PARAM, EXCLUDED = _cm.EMB_PARAM, _cm.EXCLUDED


# --------------------------------------------------------------------------
# LoRA targeting  (Task 4)
# --------------------------------------------------------------------------
# The six encoder projections an adapter can be attached to. UniMol v1's
# attention has a single fused in_proj; Task 3's converter splits it into
# q/k/v, so each is independently targetable here.
LORA_TARGETABLE = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def make_lora_flags(targets):
    """Validate a set of LoRA target names and return a {proj: bool} dict.

    The same flag set is applied to every encoder layer. The conservative
    start is ``("q_proj", "v_proj")``; expand towards all of LORA_TARGETABLE.
    """
    targets = set(targets)
    unknown = targets - set(LORA_TARGETABLE)
    if unknown:
        raise ValueError(
            f"unknown LoRA target(s) {sorted(unknown)}; "
            f"valid targets are {list(LORA_TARGETABLE)}")
    if not targets:
        raise ValueError("lora_targets is empty -- the encoder would have no "
                         "trainable adapters (only the head would train)")
    return {name: (name in targets) for name in LORA_TARGETABLE}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
class UniMolV1Config:
    """Static configuration for the `molecule` UniMol v1 architecture."""

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        n_layers=15,
        n_heads=64,
        ffn_dim=2048,
        n_kernels=128,            # K, number of Gaussian kernels
        padding_idx=1,
        r_lora=4,                 # LoRA adapter rank (capacity)
        lora_alpha=4.0,           # LoRA scaling numerator (alpha / r_lora)
        lora_targets=("q_proj", "v_proj"),
        lora_q_scale=None,        # Q init std; None -> 1/sqrt(in_dim)
        head_hidden=512,
        head_out=1,               # ESOL regression -> 1
        dtype=jnp.float32,
    ):
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.ffn_dim = ffn_dim
        self.n_kernels = n_kernels
        self.n_edge_type = vocab_size * vocab_size
        self.padding_idx = padding_idx
        self.r_lora = r_lora
        self.lora_alpha = lora_alpha
        self.lora_targets = make_lora_flags(lora_targets)   # validated flag dict
        self.lora_q_scale = lora_q_scale
        self.head_hidden = head_hidden
        self.head_out = head_out
        self.dtype = dtype


# --------------------------------------------------------------------------
# functional helpers
# --------------------------------------------------------------------------
def gaussian(x, mean, std):
    """Gaussian RBF -- bit-faithful to UniMol's torch implementation.

    Note the truncated constant ``pi = 3.14159`` (NOT ``jnp.pi``); keeping it
    is required for forward-pass parity in Task 3.
    """
    pi = 3.14159
    a = (2.0 * pi) ** 0.5
    return jnp.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


def layernorm_affine(x, weight, bias, eps=1e-5):
    """Affine LayerNorm over the last axis.

    Matches torch ``nn.LayerNorm``: biased variance, eps inside the sqrt.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)          # biased (ddof=0)
    return (x - mean) / jnp.sqrt(var + eps) * weight + bias


def gelu(x):
    """Exact (erf-based) GELU -- matches torch ``F.gelu`` default."""
    return jax.nn.gelu(x, approximate=False)


def _excluded(value):
    """Wrap a raw array as a frozen (EXCLUDED) leaf init."""
    return CommonInit(frozen_params=None, params=value, scan_map=(), es_map=EXCLUDED)


def _excluded_group(**kwargs):
    """Merge several raw arrays into one EXCLUDED sub-tree."""
    return merge_inits(**{k: _excluded(v) for k, v in kwargs.items()})


# --------------------------------------------------------------------------
# Projection: frozen W (+ optional bias) with an optional LoRA adapter
# --------------------------------------------------------------------------
def _lora_q_scale(cfg, in_dim):
    """Std for the LoRA-A (Q) init. cfg.lora_q_scale, else 1/sqrt(in_dim)."""
    if cfg.lora_q_scale is not None:
        return cfg.lora_q_scale
    return 1.0 / jnp.sqrt(in_dim)


class Projection(Model):
    """A linear projection of UniMol's encoder.

    params tree:
        W      : (out, in)   frozen pretrained weight     -- es EXCLUDED
        bias   : (out,)      frozen pretrained bias       -- es EXCLUDED  (optional)
        Q      : (r_lora, in)  LoRA down-projection       -- es MM_PARAM  (optional)
        P      : (out, r_lora) LoRA up-projection         -- es MM_PARAM  (optional)

    forward:  y = x @ W.T  +  (alpha / r_lora) * (x @ Q.T) @ P.T  +  bias

    W and bias are read directly from `params` (plain ops, never perturbed).
    Q and P go through `MM` -> `noiser.do_mm`, so EGGROLL perturbs/updates them.
    A projection without the 'Q'/'P' keys is simply a frozen linear layer.
    """

    @classmethod
    def rand_init(cls, key, W, bias, lora, r_lora, alpha, q_scale, dtype):
        out_dim, in_dim = W.shape
        inits = {"W": _excluded(jnp.asarray(W, dtype))}
        if bias is not None:
            inits["bias"] = _excluded(jnp.asarray(bias, dtype))
        if lora:
            qk, _pk = jax.random.split(key)
            Q = jax.random.normal(qk, (r_lora, in_dim), dtype) * q_scale
            P = jnp.zeros((out_dim, r_lora), dtype)          # P = 0 -> adapter starts as no-op
            inits["Q"] = CommonInit(None, Q, (), MM_PARAM)
            inits["P"] = CommonInit(None, P, (), MM_PARAM)
        merged = merge_inits(**inits)
        return merge_frozen(merged, r_lora=r_lora, alpha=alpha)

    @classmethod
    def _forward(cls, cp, x, *args, **kwargs):
        y = x @ cp.params["W"].T                              # frozen base, plain matmul
        if "Q" in cp.params:                                  # static membership check
            h = call_submodule(MM, "Q", cp, x)                # x @ Q.T  (EGGROLL-perturbed)
            d = call_submodule(MM, "P", cp, h)                # h @ P.T  (EGGROLL-perturbed)
            scale = cp.frozen_params["alpha"] / cp.frozen_params["r_lora"]
            y = y + scale * d
        if "bias" in cp.params:
            y = y + cp.params["bias"]                         # frozen bias, plain add
        return y


# --------------------------------------------------------------------------
# GaussianEdgeEncoder: 3D distance -> per-head attention bias  (fully frozen)
# --------------------------------------------------------------------------
class GaussianEdgeEncoder(Model):
    """GaussianLayer + gbf_proj (NonLinearHead). All parameters EXCLUDED.

    params (all EXCLUDED):
        mul        : (n_edge_type, 1)        edge-type scale     (nn.Embedding)
        bias       : (n_edge_type, 1)        edge-type shift     (nn.Embedding)
        means      : (1, K)                  Gaussian means
        stds       : (1, K)                  Gaussian stds
        gbf_lin1_w : (K, K)   gbf_lin1_b : (K,)
        gbf_lin2_w : (H, K)   gbf_lin2_b : (H,)

    forward(dist (L,L), edge_type (L,L) int) -> graph_attn_bias (H, L, L)
    """

    @classmethod
    def rand_init(cls, key, weights, n_kernels, dtype):
        return merge_frozen(
            _excluded_group(**{k: jnp.asarray(v, dtype) for k, v in weights.items()}),
            n_kernels=n_kernels,
        )

    @classmethod
    def _forward(cls, cp, dist, edge_type, *args, **kwargs):
        K = cp.frozen_params["n_kernels"]
        p = cp.params
        mul = p["mul"][edge_type]                             # (L, L, 1)
        bias = p["bias"][edge_type]                           # (L, L, 1)
        gx = mul * dist[..., None] + bias                     # (L, L, 1)
        gx = jnp.broadcast_to(gx, gx.shape[:-1] + (K,))       # (L, L, K)
        mean = p["means"][0]                                  # (K,)
        std = jnp.abs(p["stds"][0]) + 1e-5                    # (K,)
        g = gaussian(gx, mean, std)                           # (L, L, K)
        g = g @ p["gbf_lin1_w"].T + p["gbf_lin1_b"]           # (L, L, K)
        g = gelu(g)
        g = g @ p["gbf_lin2_w"].T + p["gbf_lin2_b"]           # (L, L, H)
        return jnp.transpose(g, (2, 0, 1))                    # (H, L, L)


# --------------------------------------------------------------------------
# MultiHeadAttention: pair-bias self-attention (single molecule)
# --------------------------------------------------------------------------
class MultiHeadAttention(Model):
    """UniMol v1 self-attention with an additive, accumulating pair bias.

    Returns (output, new_bias) where new_bias = QK^T + incoming_bias -- the
    pre-softmax logits, which become the next layer's bias (matches torch
    `return_attn=True`).
    """

    @classmethod
    def rand_init(cls, key, proj_weights, lora_flags, cfg):
        ks = jax.random.split(key, 4)
        names = ("q_proj", "k_proj", "v_proj", "out_proj")
        subs = {}
        for k, name in zip(ks, names):
            W, b = proj_weights[name]
            subs[name] = Projection.rand_init(
                k, W, b, lora=lora_flags[name],
                r_lora=cfg.r_lora, alpha=cfg.lora_alpha,
                q_scale=_lora_q_scale(cfg, W.shape[1]), dtype=cfg.dtype,
            )
        return merge_frozen(
            merge_inits(**subs),
            n_heads=cfg.n_heads, head_dim=cfg.head_dim,
        )

    @classmethod
    def _forward(cls, cp, x, bias, padding, *args, **kwargs):
        H = cp.frozen_params["n_heads"]
        hd = cp.frozen_params["head_dim"]
        L = x.shape[0]
        scaling = hd ** -0.5

        q = call_submodule(Projection, "q_proj", cp, x)       # (L, H*hd)
        k = call_submodule(Projection, "k_proj", cp, x)
        v = call_submodule(Projection, "v_proj", cp, x)

        # (L, H*hd) -> (H, L, hd)
        q = jnp.transpose(q.reshape(L, H, hd), (1, 0, 2)) * scaling
        k = jnp.transpose(k.reshape(L, H, hd), (1, 0, 2))
        v = jnp.transpose(v.reshape(L, H, hd), (1, 0, 2))

        logits = jnp.einsum("hld,hmd->hlm", q, k)             # (H, L, L) == bmm(q, k^T)
        logits = logits + bias                                # add running pair bias
        attn = jax.nn.softmax(logits, axis=-1)                # over keys
        o = jnp.einsum("hlm,hmd->hld", attn, v)               # (H, L, hd)
        o = jnp.transpose(o, (1, 0, 2)).reshape(L, H * hd)    # (L, H*hd)
        o = call_submodule(Projection, "out_proj", cp, o)
        return o, logits                                      # logits -> next layer's bias


# --------------------------------------------------------------------------
# EncoderLayer: pre-LN transformer block with pair bias
# --------------------------------------------------------------------------
class EncoderLayer(Model):
    """One UniMol v1 transformer encoder layer (pre-LN, dropout removed)."""

    @classmethod
    def rand_init(cls, key, layer_weights, lora_flags, cfg):
        attn_key, _ = jax.random.split(key)
        proj_weights = {n: layer_weights[n] for n in
                        ("q_proj", "k_proj", "v_proj", "out_proj")}
        subs = dict(
            attn=MultiHeadAttention.rand_init(attn_key, proj_weights, lora_flags, cfg),
            self_attn_layer_norm=_excluded_group(
                weight=jnp.asarray(layer_weights["ln1_w"], cfg.dtype),
                bias=jnp.asarray(layer_weights["ln1_b"], cfg.dtype)),
            final_layer_norm=_excluded_group(
                weight=jnp.asarray(layer_weights["ln2_w"], cfg.dtype),
                bias=jnp.asarray(layer_weights["ln2_b"], cfg.dtype)),
        )
        for fc in ("fc1", "fc2"):
            W, b = layer_weights[fc]
            subs[fc] = Projection.rand_init(
                key, W, b, lora=lora_flags[fc],
                r_lora=cfg.r_lora, alpha=cfg.lora_alpha,
                q_scale=_lora_q_scale(cfg, W.shape[1]), dtype=cfg.dtype,
            )
        return merge_inits(**subs)

    @classmethod
    def _forward(cls, cp, x, bias, padding, *args, **kwargs):
        ln1 = cp.params["self_attn_layer_norm"]
        ln2 = cp.params["final_layer_norm"]

        residual = x
        h = layernorm_affine(x, ln1["weight"], ln1["bias"])
        o, new_bias = call_submodule(MultiHeadAttention, "attn", cp, h, bias, padding)
        x = residual + o

        residual = x
        h = layernorm_affine(x, ln2["weight"], ln2["bias"])
        h = call_submodule(Projection, "fc1", cp, h)
        h = gelu(h)
        h = call_submodule(Projection, "fc2", cp, h)
        x = residual + h
        return x, new_bias


# --------------------------------------------------------------------------
# EncoderStack: the N encoder layers (carry x and the accumulating bias)
# --------------------------------------------------------------------------
class EncoderStack(Model):
    """Sequence of EncoderLayers, looped Python-style (cf. EGGROLL `MLP`)."""

    @classmethod
    def rand_init(cls, key, layers_weights, lora_flags, cfg):
        keys = jax.random.split(key, len(layers_weights))
        subs = {
            str(i): EncoderLayer.rand_init(keys[i], layers_weights[i], lora_flags, cfg)
            for i in range(len(layers_weights))
        }
        return merge_inits(**subs)

    @classmethod
    def _forward(cls, cp, x, bias, padding, *args, **kwargs):
        n = len(cp.params)
        for i in range(n):
            x, bias = call_submodule(EncoderLayer, str(i), cp, x, bias, padding)
        return x, bias


# --------------------------------------------------------------------------
# Encoder: emb LayerNorm -> stack -> final LayerNorm
# --------------------------------------------------------------------------
class Encoder(Model):
    """TransformerEncoderWithPair (UniMol v1)."""

    @classmethod
    def rand_init(cls, key, encoder_weights, lora_flags, cfg):
        return merge_inits(
            emb_layer_norm=_excluded_group(
                weight=jnp.asarray(encoder_weights["emb_ln_w"], cfg.dtype),
                bias=jnp.asarray(encoder_weights["emb_ln_b"], cfg.dtype)),
            final_layer_norm=_excluded_group(
                weight=jnp.asarray(encoder_weights["final_ln_w"], cfg.dtype),
                bias=jnp.asarray(encoder_weights["final_ln_b"], cfg.dtype)),
            layers=EncoderStack.rand_init(key, encoder_weights["layers"], lora_flags, cfg),
        )

    @classmethod
    def _forward(cls, cp, emb, graph_attn_bias, padding, *args, **kwargs):
        eln = cp.params["emb_layer_norm"]
        fln = cp.params["final_layer_norm"]

        x = layernorm_affine(emb, eln["weight"], eln["bias"])
        x = jnp.where(padding[:, None], 0.0, x)               # zero padded tokens

        # merge padding into the initial pair bias: padded *keys* -> -inf,
        # which then propagates through every layer (matches `fill_attn_mask`).
        neg_inf = jnp.asarray(-jnp.inf, graph_attn_bias.dtype)
        bias = jnp.where(padding[None, None, :], neg_inf, graph_attn_bias)

        x, _bias = call_submodule(EncoderStack, "layers", cp, x, bias, padding)
        x = layernorm_affine(x, fln["weight"], fln["bias"])
        return x


# --------------------------------------------------------------------------
# RegressionHead: new MLP head for ESOL, trained from scratch
# --------------------------------------------------------------------------
class RegressionHead(Model):
    """dense -> tanh -> out.  weights MM_PARAM, biases PARAM (trained by EGGROLL)."""

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dim, out_dim, dtype):
        dk, ok = jax.random.split(key)
        return merge_inits(
            dense=Linear.rand_init(dk, in_dim, hidden_dim, use_bias=True, dtype=dtype),
            out=Linear.rand_init(ok, hidden_dim, out_dim, use_bias=True, dtype=dtype),
        )

    @classmethod
    def _forward(cls, cp, x, *args, **kwargs):
        h = call_submodule(Linear, "dense", cp, x)
        h = jnp.tanh(h)
        out = call_submodule(Linear, "out", cp, h)
        return out[0] if out.shape == (1,) else out


# --------------------------------------------------------------------------
# UniMolV1: the full model
# --------------------------------------------------------------------------
class UniMolV1(Model):
    """UniMol v1 (`molecule`) forward pass for one molecule.

    forward(src_tokens (L,), src_distance (L,L), src_edge_type (L,L))
        -> scalar prediction (ESOL log-solubility)
    """

    @classmethod
    def rand_init(cls, key, weights, cfg):
        """Assemble the EGGROLL parameter tree.

        `weights` is the converted-checkpoint pytree produced by Task 3:
            weights['embed_tokens']            : (vocab, embed_dim)
            weights['gaussian']                : dict of GaussianEdgeEncoder tables
            weights['encoder']                 : dict with emb_ln/final_ln/layers
            weights['encoder']['layers'][i]    : dict per layer (see make_random_weights)
        The head is initialised from scratch here (not from the checkpoint).
        """
        enc_key, head_key = jax.random.split(key)

        # which projections carry a LoRA adapter (validated flag dict,
        # the same set for every encoder layer)
        lora_flags = cfg.lora_targets

        merged = merge_inits(
            embed_tokens=_excluded(jnp.asarray(weights["embed_tokens"], cfg.dtype)),
            gaussian=GaussianEdgeEncoder.rand_init(
                key, weights["gaussian"], cfg.n_kernels, cfg.dtype),
            encoder=Encoder.rand_init(enc_key, weights["encoder"], lora_flags, cfg),
            head=RegressionHead.rand_init(
                head_key, cfg.embed_dim, cfg.head_hidden, cfg.head_out, cfg.dtype),
        )
        return merge_frozen(merged, padding_idx=cfg.padding_idx)

    @classmethod
    def _forward(cls, cp, src_tokens, src_distance, src_edge_type,
                 return_repr=False, *args, **kwargs):
        """If return_repr is True, return the CLS representation (the frozen
        encoder output) instead of the head prediction -- used for Task 3
        parity verification against the PyTorch reference."""
        padding = (src_tokens == cp.frozen_params["padding_idx"])     # (L,) bool

        emb = cp.params["embed_tokens"][src_tokens]                   # (L, embed_dim)
        graph_attn_bias = call_submodule(
            GaussianEdgeEncoder, "gaussian", cp, src_distance, src_edge_type)  # (H, L, L)

        enc = call_submodule(Encoder, "encoder", cp, emb, graph_attn_bias, padding)
        cls_repr = enc[0]                                             # CLS token (L=0)
        if return_repr:
            return cls_repr
        return call_submodule(RegressionHead, "head", cp, cls_repr)   # scalar


# --------------------------------------------------------------------------
# utility: fabricate a correctly-shaped `weights` pytree (for testing /
# scaffolding before Task 3's real checkpoint converter exists)
# --------------------------------------------------------------------------
def make_random_weights(key, cfg):
    """Random `weights` pytree with the exact structure Task 3 must produce."""
    ed, ff = cfg.embed_dim, cfg.ffn_dim
    K, H, V = cfg.n_kernels, cfg.n_heads, cfg.vocab_size

    def rn(k, shape, scale=0.02):
        return jax.random.normal(k, shape, cfg.dtype) * scale

    keys = iter(jax.random.split(key, 8 + cfg.n_layers * 16))

    def proj(out_dim, in_dim):
        return (rn(next(keys), (out_dim, in_dim)), rn(next(keys), (out_dim,)))

    layers = []
    for _ in range(cfg.n_layers):
        layers.append(dict(
            q_proj=proj(ed, ed), k_proj=proj(ed, ed),
            v_proj=proj(ed, ed), out_proj=proj(ed, ed),
            fc1=proj(ff, ed), fc2=proj(ed, ff),
            ln1_w=jnp.ones(ed, cfg.dtype), ln1_b=jnp.zeros(ed, cfg.dtype),
            ln2_w=jnp.ones(ed, cfg.dtype), ln2_b=jnp.zeros(ed, cfg.dtype),
        ))

    return dict(
        embed_tokens=rn(next(keys), (V, ed)),
        gaussian=dict(
            mul=jnp.ones((cfg.n_edge_type, 1), cfg.dtype),
            bias=jnp.zeros((cfg.n_edge_type, 1), cfg.dtype),
            means=jax.random.uniform(next(keys), (1, K), cfg.dtype, 0.0, 3.0),
            stds=jax.random.uniform(next(keys), (1, K), cfg.dtype, 0.0, 3.0),
            gbf_lin1_w=rn(next(keys), (K, K)), gbf_lin1_b=jnp.zeros(K, cfg.dtype),
            gbf_lin2_w=rn(next(keys), (H, K)), gbf_lin2_b=jnp.zeros(H, cfg.dtype),
        ),
        encoder=dict(
            emb_ln_w=jnp.ones(ed, cfg.dtype), emb_ln_b=jnp.zeros(ed, cfg.dtype),
            final_ln_w=jnp.ones(ed, cfg.dtype), final_ln_b=jnp.zeros(ed, cfg.dtype),
            layers=layers,
        ),
    )

# --------------------------------------------------------------------------
# Task 3: PyTorch checkpoint -> JAX `weights` pytree
# --------------------------------------------------------------------------
import numpy as _np


def _to_array(t):
    """Convert a torch.Tensor (or ndarray-like) to a numpy array.

    The source dtype is preserved; the final cast to the target dtype is
    done by ``transform_torch_model`` via ``jnp.asarray(..., dtype)``.
    Handles the real pretrained checkpoint (torch tensors -> .numpy()) and
    plain numpy arrays alike, so the converter is testable without torch.
    """
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    return _np.asarray(t)


def load_torch_checkpoint(path):
    """Load a UniMol v1 ``.pt`` checkpoint and return the flat state_dict.

    Mirrors ``UniMolModel.load_pretrained_weights``: unwraps the optional
    ``model`` / ``model_state_dict`` container key.
    """
    import torch
    sd = torch.load(path, map_location=lambda storage, loc: storage)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    return sd


def transform_torch_model(state_dict, cfg, dtype=jnp.float32):
    """Convert a UniMol v1 PyTorch ``state_dict`` into the nested ``weights``
    pytree consumed by :meth:`UniMolV1.rand_init`.

    Key transformations:

      * The fused attention projection is **split** (the Task-1 decision)::

            self_attn.in_proj.weight  (3*ed, ed)  ->  q/k/v  each (ed, ed)
            self_attn.in_proj.bias    (3*ed,)     ->  q/k/v  each (ed,)

        Row blocks follow ``chunk(3, dim=-1)`` of the projection *output*:
        rows [0:ed]=q, [ed:2ed]=k, [2ed:3ed]=v.

      * torch tensors -> ``jnp.array`` via ``.numpy()`` (see ``_to_array``).

      * ``classification_head.*`` is intentionally **not** converted -- the
        ESOL head is trained from scratch by EGGROLL.

    The output tree is structurally identical to ``make_random_weights`` so
    that ``UniMolV1.rand_init`` accepts it directly.
    """
    sd = state_dict
    ed = cfg.embed_dim

    def arr(key):
        if key not in sd:
            raise KeyError(f"transform_torch_model: missing checkpoint key '{key}'")
        return jnp.asarray(_to_array(sd[key]), dtype)

    def proj(prefix):
        return (arr(prefix + ".weight"), arr(prefix + ".bias"))

    layers = []
    for i in range(cfg.n_layers):
        p = f"encoder.layers.{i}."
        W_in = _to_array(sd[p + "self_attn.in_proj.weight"])      # (3*ed, ed)
        b_in = _to_array(sd[p + "self_attn.in_proj.bias"])        # (3*ed,)
        if W_in.shape != (3 * ed, ed):
            raise ValueError(
                f"layer {i}: in_proj.weight has shape {W_in.shape}, "
                f"expected {(3 * ed, ed)}")
        layers.append(dict(
            q_proj=(jnp.asarray(W_in[0:ed], dtype), jnp.asarray(b_in[0:ed], dtype)),
            k_proj=(jnp.asarray(W_in[ed:2 * ed], dtype), jnp.asarray(b_in[ed:2 * ed], dtype)),
            v_proj=(jnp.asarray(W_in[2 * ed:3 * ed], dtype), jnp.asarray(b_in[2 * ed:3 * ed], dtype)),
            out_proj=proj(p + "self_attn.out_proj"),
            fc1=proj(p + "fc1"),
            fc2=proj(p + "fc2"),
            ln1_w=arr(p + "self_attn_layer_norm.weight"),
            ln1_b=arr(p + "self_attn_layer_norm.bias"),
            ln2_w=arr(p + "final_layer_norm.weight"),
            ln2_b=arr(p + "final_layer_norm.bias"),
        ))

    return dict(
        embed_tokens=arr("embed_tokens.weight"),
        gaussian=dict(
            mul=arr("gbf.mul.weight"),
            bias=arr("gbf.bias.weight"),
            means=arr("gbf.means.weight"),
            stds=arr("gbf.stds.weight"),
            gbf_lin1_w=arr("gbf_proj.linear1.weight"),
            gbf_lin1_b=arr("gbf_proj.linear1.bias"),
            gbf_lin2_w=arr("gbf_proj.linear2.weight"),
            gbf_lin2_b=arr("gbf_proj.linear2.bias"),
        ),
        encoder=dict(
            emb_ln_w=arr("encoder.emb_layer_norm.weight"),
            emb_ln_b=arr("encoder.emb_layer_norm.bias"),
            final_ln_w=arr("encoder.final_layer_norm.weight"),
            final_ln_b=arr("encoder.final_layer_norm.bias"),
            layers=layers,
        ),
    )

# --------------------------------------------------------------------------
# Task 4: es_map audit & validation
# --------------------------------------------------------------------------
# The es_map is built automatically by `merge_inits` while assembling the
# parameter tree; every leaf carries one of PARAM / MM_PARAM / EMB_PARAM /
# EXCLUDED. The helpers below make the resulting optimisation set theta
# explicit and verify the intended partition:
#
#     theta = { all P, Q LoRA adapters in the encoder }  (MM_PARAM)
#           u { head weights (MM_PARAM), head biases (PARAM) }
#
# everything else -- embed_tokens, the Gaussian tables, every frozen W, all
# encoder biases and LayerNorms -- is EXCLUDED. Note PARAM is used ONLY for
# the head biases: that is why the EGGROLL noiser must run with
# freeze_nonlora=False (freeze_nonlora=True would also freeze the head bias;
# encoder biases/LayerNorms are frozen via EXCLUDED instead).

_ES_NAME = {PARAM: "PARAM", MM_PARAM: "MM_PARAM",
            EMB_PARAM: "EMB_PARAM", EXCLUDED: "EXCLUDED"}


def _path_parts(path):
    """jax key-path -> tuple of plain strings."""
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return tuple(parts)


def _expected_tag(parts):
    """The es_map tag a leaf at this path *should* carry."""
    tail = parts[-1]
    if tail == "W":                       # Projection frozen base weight
        return EXCLUDED
    if tail in ("P", "Q"):                # LoRA adapter -> EGGROLL low-rank
        return MM_PARAM
    if "head" in parts:                   # trained-from-scratch ESOL head
        if tail == "weight":
            return MM_PARAM
        if tail == "bias":
            return PARAM
    return EXCLUDED                       # embed / gaussian / LN / proj bias


def theta_summary(params, es_map, verbose=True):
    """Audit the assembled tree: report the optimisation set theta.

    Returns a dict; pretty-prints it when verbose. Categorises every leaf as
    frozen (EXCLUDED), an encoder LoRA adapter, or a head parameter.
    """
    p_leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    e_leaves = jax.tree_util.tree_flatten_with_path(es_map)[0]
    if jax.tree.structure(params) != jax.tree.structure(es_map):
        raise ValueError("params and es_map have different tree structures")

    rep = dict(total_elements=0,
               frozen=dict(leaves=0, elements=0),
               lora=dict(leaves=0, elements=0, paths=[]),
               head=dict(leaves=0, elements=0, paths=[]))
    for (path, leaf), (_, tag) in zip(p_leaves, e_leaves):
        parts = _path_parts(path)
        n = int(jax.numpy.asarray(leaf).size)
        rep["total_elements"] += n
        if tag == EXCLUDED:
            rep["frozen"]["leaves"] += 1
            rep["frozen"]["elements"] += n
        elif "head" in parts:
            rep["head"]["leaves"] += 1
            rep["head"]["elements"] += n
            rep["head"]["paths"].append((".".join(parts), _ES_NAME[tag], n))
        else:
            rep["lora"]["leaves"] += 1
            rep["lora"]["elements"] += n
            rep["lora"]["paths"].append((".".join(parts), _ES_NAME[tag], n))

    trainable = rep["lora"]["elements"] + rep["head"]["elements"]
    rep["trainable_elements"] = trainable
    rep["trainable_fraction"] = trainable / max(rep["total_elements"], 1)

    if verbose:
        print("theta (optimisation set) summary")
        print(f"  total parameters    : {rep['total_elements']:,}")
        print(f"  frozen  (EXCLUDED)  : {rep['frozen']['elements']:,} "
              f"in {rep['frozen']['leaves']} leaves")
        print(f"  encoder LoRA P,Q    : {rep['lora']['elements']:,} "
              f"in {rep['lora']['leaves']} leaves")
        print(f"  head weights+biases : {rep['head']['elements']:,} "
              f"in {rep['head']['leaves']} leaves")
        print(f"  trainable theta     : {trainable:,} "
              f"({100 * rep['trainable_fraction']:.3f}% of total)")
    return rep


def validate_assembly(params, es_map):
    """Verify the es_map matches the intended partition; raise on violation.

    Checks: tree-structures agree; every leaf carries the tag implied by its
    role (W -> EXCLUDED, P/Q -> MM_PARAM, head weight -> MM_PARAM, head bias
    -> PARAM, everything else -> EXCLUDED); and PARAM appears ONLY at the
    head biases (the freeze_nonlora=False invariant).
    """
    if jax.tree.structure(params) != jax.tree.structure(es_map):
        raise AssertionError("validate_assembly: params/es_map structure mismatch")

    p_leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    e_leaves = jax.tree_util.tree_flatten_with_path(es_map)[0]

    bad = []
    param_leaves_outside_head = []
    for (path, _leaf), (_, tag) in zip(p_leaves, e_leaves):
        parts = _path_parts(path)
        want = _expected_tag(parts)
        if tag != want:
            bad.append((".".join(parts), _ES_NAME.get(tag, tag),
                        _ES_NAME[want]))
        if tag == PARAM and "head" not in parts:
            param_leaves_outside_head.append(".".join(parts))

    msg = []
    if bad:
        msg.append("es_map tag mismatches (path: got -> expected):")
        msg += [f"  {p}: {g} -> {w}" for p, g, w in bad]
    if param_leaves_outside_head:
        msg.append("PARAM leaves found outside the head -- these would be "
                   "FROZEN if freeze_nonlora=True; mark them EXCLUDED instead:")
        msg += [f"  {p}" for p in param_leaves_outside_head]
    if msg:
        raise AssertionError("validate_assembly failed:\n" + "\n".join(msg))
    return True
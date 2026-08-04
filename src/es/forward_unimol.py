"""UniMol v1 split into a frozen prefix and an ES-perturbed suffix.

Two facts about the stock encoder drive this module, both verified against
``unimol_tools/models/transformers.py``:

1. ``attn_bias`` is *stateful across layers*. ``TransformerEncoderWithPair``
   calls each layer with ``return_attn=True`` and re-binds the returned
   pre-softmax scores as the next layer's bias, i.e.
   ``bias_{l+1} = Q_l K_l^T + bias_l``. Any split of the stack therefore has to
   carry the accumulated bias across the boundary, not just the hidden state.
2. ``fill_attn_mask`` folds padding into that bias as ``-inf`` and then sets
   ``padding_mask = None``, so the per-layer attention never uses a separate key
   padding mask. The suffix only needs the bias.

The prefix is identical for every population member, so it is computed **once
per ES step and shared** -- 11 frozen layers over ``B`` molecules against 4
perturbed layers over ``N x B``, i.e. about 1% of the step. That is cheaper and
far simpler than caching it, which would mean holding an
``(n_mol, heads, S, S)`` bias tensor (~0.6-2.3 GB for ESOL).

The suffix applies EGGROLL's batched low-rank trick to every perturbed matrix:

    x (M + sigma/sqrt(r) A B^T)^T  =  x M^T + sigma/sqrt(r) (x B) A^T

so the bulk of the work stays a single high-arithmetic-intensity GEMM shared by
the whole population, with one cheap rank-r correction per member.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

#: Matrices inside a transformer block that ES may perturb.
MATRIX_NAMES = ('in_proj', 'out_proj', 'fc1', 'fc2')
#: Those whose *output* is the 512-wide residual stream, and therefore the only
#: ones where masking A by head region is meaningful (see ``ESSpec.region_mask``).
OUTPUT_SIDE = ('out_proj', 'fc2')


@dataclass
class ESSpec:
    """Which parameters ES searches over, and how perturbations are shaped."""

    #: Perturb the last ``n_layers`` transformer blocks; the rest stay frozen.
    n_layers: int = 4
    #: EGGROLL rank. The paper reports r=1 works, but every experiment there has
    #: ``N*r > min(m, n)`` so the *aggregate* update is still full rank. Here
    #: min(m, n) = 512, so N=256 needs r >= 2 to stay in that regime.
    rank: int = 4
    matrices: tuple[str, ...] = MATRIX_NAMES
    #: Restrict each A column to one head region, so a perturbation moves
    #: exactly one region of the 512-d embedding and ES credit assignment
    #: becomes "which region should change". Output-side matrices only.
    region_mask: bool = False
    n_regions: int = 16
    #: Scale sigma per matrix by ``||W||_F / sqrt(m n)``, so one global sigma
    #: means the same *relative* perturbation for a 512x512 and a 512x2048
    #: matrix. Assumption 4 of the paper wants ||mu|| = O(1); relative scaling
    #: is the practical way to get there on pretrained weights.
    relative_sigma: bool = True

    def targets(self, n_total_layers: int) -> list[tuple[int, str]]:
        """(layer index, matrix name) pairs this spec perturbs."""
        start = max(0, n_total_layers - self.n_layers)
        return [
            (li, name)
            for li in range(start, n_total_layers)
            for name in self.matrices
        ]


@dataclass
class Perturbation:
    """Low-rank factors for one population chunk, keyed by (layer, matrix)."""

    factors: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    #: Per-matrix sigma actually applied, including relative scaling.
    sigma: dict[tuple[int, str], float] = field(default_factory=dict)

    def get(self, key: tuple[int, str]):
        return self.factors.get(key)

    def slice(self, start: int, stop: int) -> 'Perturbation':
        """A view of members ``[start, stop)`` for one forward-pass tile.

        Views, not copies: the whole population's factors stay resident so the
        update can aggregate them in one pass once every chunk has been scored.
        """
        return Perturbation(
            {k: (a[start:stop], b[start:stop]) for k, (a, b) in self.factors.items()},
            dict(self.sigma),
        )


def _plinear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
             factors, sigma: float) -> torch.Tensor:
    """``x W^T + b`` plus a per-member rank-r correction.

    ``x`` is ``(N, M, in)``. The base term is one GEMM shared across the whole
    population; the correction contracts through the rank-r bottleneck, which is
    what keeps arithmetic intensity high instead of degenerating into a batched
    matrix-vector product per member.
    """
    y = F.linear(x, weight, bias)
    if factors is None:
        return y
    a, b = factors                                   # (N, out, r), (N, in, r)
    xb = torch.einsum('nmi,nir->nmr', x, b)
    return y + sigma * torch.einsum('nmr,nor->nmo', xb, a)


class SplitUniMol:
    """Run UniMol v1 as ``frozen prefix -> ES-perturbed suffix -> CLS``.

    Args:
        model: a stock ``UniMolModel`` in eval mode.
        spec: which matrices ES perturbs.
        dtype: compute dtype for the suffix. fp32 is the default because the
            sigma=0 equivalence check against the stock model has to be exact
            to fp32 tolerance; bf16 halves memory once that check has passed.
    """

    def __init__(self, model, spec: ESSpec, dtype: torch.dtype = torch.float32):
        self.model = model
        self.spec = spec
        self.dtype = dtype
        self.encoder = model.encoder
        self.layers = model.encoder.layers
        self.n_layers = len(self.layers)
        self.split_at = max(0, self.n_layers - spec.n_layers)
        self.heads = model.args.encoder_attention_heads
        self.embed_dim = model.args.encoder_embed_dim
        self.head_dim = self.embed_dim // self.heads
        self.scaling = self.head_dim ** -0.5

        self.targets = spec.targets(self.n_layers)
        self.base: dict[tuple[int, str], torch.Tensor] = {}
        self.sigma_scale: dict[tuple[int, str], float] = {}
        for key in self.targets:
            w = self._weight(key)
            self.base[key] = w.detach().clone()
            m, n = w.shape
            self.sigma_scale[key] = (
                float(w.norm()) / (m * n) ** 0.5 if spec.relative_sigma else 1.0
            )

    # --- parameter access ----------------------------------------------------

    def _module(self, layer_idx: int, name: str):
        layer = self.layers[layer_idx]
        if name in ('in_proj', 'out_proj'):
            return getattr(layer.self_attn, name)
        return getattr(layer, name)

    def _weight(self, key: tuple[int, str]) -> torch.Tensor:
        return self._module(*key).weight

    def shapes(self) -> dict[tuple[int, str], tuple[int, int]]:
        return {k: tuple(self.base[k].shape) for k in self.targets}

    def mean(self, key: tuple[int, str]) -> torch.Tensor:
        """The ES mean matrix M currently installed in the live model."""
        return self._weight(key).data

    @torch.no_grad()
    def set_mean(self, key: tuple[int, str], value: torch.Tensor) -> None:
        self._weight(key).data.copy_(value)

    def n_es_params(self) -> int:
        return sum(int(w.numel()) for w in self.base.values())

    # --- frozen prefix -------------------------------------------------------

    @torch.no_grad()
    def prefix(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed + attention bias + the frozen leading layers.

        Mirrors ``UniMolModel.forward`` and ``TransformerEncoderWithPair.forward``
        with dropout disabled. Returns ``(x, bias)`` where ``x`` is
        ``(B, S, D)`` and ``bias`` is ``(B*H, S, S)`` with padding already folded
        in as ``-inf``.
        """
        model = self.model
        src_tokens = batch['src_tokens']
        padding_mask = src_tokens.eq(model.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = model.embed_tokens(src_tokens)
        n_node = batch['src_distance'].size(-1)
        gbf_feature = model.gbf(batch['src_distance'], batch['src_edge_type'])
        bias = model.gbf_proj(gbf_feature)
        bias = bias.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        x = self.encoder.emb_layer_norm(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            # fill_attn_mask: padding becomes -inf in the bias, after which the
            # stock code drops padding_mask entirely.
            bias = bias.view(x.size(0), -1, n_node, n_node)
            bias = bias.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float('-inf')
            )
            bias = bias.view(-1, n_node, n_node)

        for li in range(self.split_at):
            x, bias, _ = self.layers[li](
                x, padding_mask=None, attn_bias=bias, return_attn=True
            )
        return x.to(self.dtype), bias.to(self.dtype)

    # --- perturbed suffix ----------------------------------------------------

    @torch.no_grad()
    def suffix(self, x0: torch.Tensor, bias0: torch.Tensor, n_members: int,
               pert: Perturbation | None = None) -> torch.Tensor:
        """Run the trailing layers for ``n_members`` perturbed copies.

        Args:
            x0: ``(B, S, D)`` shared prefix hidden state.
            bias0: ``(B*H, S, S)`` shared accumulated attention bias.
            n_members: population chunk size N.
            pert: low-rank factors; ``None`` runs the unperturbed model, which
                must reproduce the stock CLS representation exactly.

        Returns:
            ``(N, B, D)`` CLS representations.
        """
        n_mol, seq, dim = x0.shape
        n = n_members
        bh = n_mol * self.heads

        x = x0.unsqueeze(0).expand(n, n_mol, seq, dim).reshape(n, n_mol * seq, dim)
        bias = bias0  # (B*H, S, S) until the first layer makes it per-member

        for li in range(self.split_at, self.n_layers):
            layer = self.layers[li]
            residual = x
            h = layer.self_attn_layer_norm(x)

            attn = layer.self_attn
            qkv = _plinear(h, attn.in_proj.weight, attn.in_proj.bias,
                           pert.get((li, 'in_proj')) if pert else None,
                           pert.sigma.get((li, 'in_proj'), 0.0) if pert else 0.0)
            q, k, v = qkv.chunk(3, dim=-1)
            q = self._to_heads(q, n, n_mol, seq) * self.scaling
            k = self._to_heads(k, n, n_mol, seq)
            v = self._to_heads(v, n, n_mol, seq)

            scores = torch.bmm(q, k.transpose(1, 2)).view(n, bh, seq, seq)
            # First perturbed layer: bias is still shared, so broadcast-add it
            # rather than materialising N copies. Afterwards it is per-member.
            scores = scores + (bias.unsqueeze(0) if bias.dim() == 3 else bias)
            bias = scores
            probs = torch.softmax(scores.float(), dim=-1).to(self.dtype)

            o = torch.bmm(probs.view(-1, seq, seq), v)
            o = o.view(n, n_mol, self.heads, seq, self.head_dim)
            o = o.permute(0, 1, 3, 2, 4).reshape(n, n_mol * seq, dim)
            o = _plinear(o, attn.out_proj.weight, attn.out_proj.bias,
                         pert.get((li, 'out_proj')) if pert else None,
                         pert.sigma.get((li, 'out_proj'), 0.0) if pert else 0.0)
            x = residual + o

            residual = x
            h = layer.final_layer_norm(x)
            h = _plinear(h, layer.fc1.weight, layer.fc1.bias,
                         pert.get((li, 'fc1')) if pert else None,
                         pert.sigma.get((li, 'fc1'), 0.0) if pert else 0.0)
            h = layer.activation_fn(h)
            h = _plinear(h, layer.fc2.weight, layer.fc2.bias,
                         pert.get((li, 'fc2')) if pert else None,
                         pert.sigma.get((li, 'fc2'), 0.0) if pert else 0.0)
            x = residual + h

        x = self.encoder.final_layer_norm(x)
        return x.view(n, n_mol, seq, dim)[:, :, 0, :]

    def _to_heads(self, t: torch.Tensor, n: int, n_mol: int,
                  seq: int) -> torch.Tensor:
        """``(N, B*S, D) -> (N*B*H, S, hd)``, matching the stock reshape order.

        ``reshape`` rather than ``view``: the input is a ``chunk()`` slice of the
        fused QKV projection, so it is strided along the feature axis.
        """
        t = t.reshape(n, n_mol, seq, self.heads, self.head_dim)
        return t.permute(0, 1, 3, 2, 4).reshape(-1, seq, self.head_dim)

    # --- diagnostics ---------------------------------------------------------

    def attn_bytes(self, n_members: int, n_mol: int, seq: int) -> int:
        """Bytes for one attention-score tensor -- the memory driver.

        Peak is roughly 3x this (running bias, fresh scores, softmax output),
        and it grows with ``N * B * S^2``, so shrinking the molecule tile is the
        cheapest knob when a chunk does not fit.
        """
        return (
            n_members * n_mol * self.heads * seq * seq
            * torch.finfo(self.dtype).bits // 8
        )

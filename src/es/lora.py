"""
LoRA utilities for UniMol-GP Step 2 (LoRA-subspace adaptation).

Design goals
------------
* Wrap every ``nn.Linear`` inside the (frozen) UniMol transformer encoder with a
  low-rank adapter ``W = W0 + (alpha/r) * B @ A`` where ``B`` is zero-initialised
  so that **at step 0 the model is exactly the pretrained backbone**.
* Keep the backbone frozen; only the LoRA ``A/B`` matrices (+ the regression head)
  are trainable. This is the search space for both Step 2.1 (GD) and Step 2.2 (ES).
* After training, ``merge_lora_`` folds the adapters back into plain ``nn.Linear``
  layers so the resulting ``state_dict`` is **architecturally identical to a normal
  fine-tuned UniMol** -> Step-1's evaluation path (MolPredict) can load it as-is.

This file is pure PyTorch and has no dependency on unimol_tools.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that adds a low-rank adapter.

    forward(x) = base(x) + scaling * (x @ A^T) @ B^T
    where A in R[r, in], B in R[out, r], scaling = alpha / r.
    The wrapped ``base`` layer is frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)

        # LoRA params live in fp32 regardless of base dtype (ES/Adam operate in fp32).
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B = 0  ->  delta_W = 0  ->  model == pretrained at start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # Cast LoRA matmul to the activation dtype (handles autocast bf16/fp16 cleanly).
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        delta = F.linear(F.linear(x, a), b) * self.scaling
        return out + delta

    @torch.no_grad()
    def to_merged_linear(self) -> nn.Linear:
        """Return a plain ``nn.Linear`` with the adapter folded into the weight."""
        merged = nn.Linear(self.in_features, self.out_features,
                            bias=self.base.bias is not None)
        merged = merged.to(self.base.weight.device, self.base.weight.dtype)
        delta = (self.scaling * (self.lora_B @ self.lora_A)).to(self.base.weight.dtype)
        merged.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def inject_lora_(module: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Recursively replace every ``nn.Linear`` child of ``module`` with ``LoRALinear``.

    Returns the number of wrapped layers. Call this on ``model.encoder`` so that
    distance featurisation (``model.gbf``, ``model.gbf_proj``), token embeddings and
    the head are left untouched.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += inject_lora_(child, rank=rank, alpha=alpha)
    return n


def merge_lora_(module: nn.Module) -> int:
    """Recursively fold every ``LoRALinear`` back into a plain ``nn.Linear`` (in place)."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.to_merged_linear())
            n += 1
        else:
            n += merge_lora_(child)
    return n


def setup_trainable_(model: nn.Module, train_head: bool = True,
                     head_attr: str = "classification_head") -> None:
    """Freeze everything, then unfreeze LoRA A/B (and optionally the regression head)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    if train_head and hasattr(model, head_attr):
        for p in getattr(model, head_attr).parameters():
            p.requires_grad_(True)


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_params(params: Iterable[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
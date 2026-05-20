"""
Helper to enumerate LoRA target parameter names for Unimol-v1 encoder.

For Unimol-v1 (molecule), structure is:
    encoder.layers.{0..14}.self_attn.{q_proj,k_proj,v_proj,out_proj}.weight  (4 × 15 = 60)
    encoder.layers.{0..14}.fc1.weight                                        (15)
    encoder.layers.{0..14}.fc2.weight                                        (15)
    Total: 90 Linear weights

We do NOT include:
    - LayerNorm params (small, often kept FULL or frozen)
    - embed_tokens.weight (vocab embedding, usually frozen for fine-tune)
    - gbf, gbf_proj (special distance encoding, sensitive init)
    - classification_head (we don't use it — ridge regression replaces it)
"""
from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn


def get_unimol_encoder_lora_targets(
    model: nn.Module,
    include_q: bool = True,
    include_k: bool = True,
    include_v: bool = True,
    include_out: bool = True,
    include_fc1: bool = True,
    include_fc2: bool = True,
    layer_filter: List[int] | None = None,
) -> List[str]:
    """Return list of full param names (with .weight suffix) for LoRA-targeted Linears.

    Args:
        layer_filter: if given, only include these layer indices (e.g. [0, 1, 14]).
                      None = all layers.
    """
    keywords = []
    if include_q:    keywords.append("q_proj")
    if include_k:    keywords.append("k_proj")
    if include_v:    keywords.append("v_proj")
    if include_out:  keywords.append("out_proj")
    if include_fc1:  keywords.append("fc1")
    if include_fc2:  keywords.append("fc2")

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "encoder.layers." not in name:
            continue
        # Extract layer index
        try:
            after = name.split("encoder.layers.")[1]
            idx_str = after.split(".")[0]
            layer_idx = int(idx_str)
        except (IndexError, ValueError):
            continue
        if layer_filter is not None and layer_idx not in layer_filter:
            continue
        # Match keyword
        if not any(kw in name.split(".")[-1] or kw == name.split(".")[-1] for kw in keywords):
            # More precise: check if the last component matches
            last = name.split(".")[-1]
            if last not in keywords:
                continue
        targets.append(name + ".weight")
    return sorted(set(targets))


def print_unimol_structure(model: nn.Module, max_lines: int = 80):
    """Print Unimol model structure for inspection."""
    print("=" * 70)
    print(" Unimol model structure (Linear layers with shapes)")
    print("=" * 70)
    total_params = 0
    n_lines = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            total_params += params
            if n_lines < max_lines:
                print(f"  Linear  {name:60s}  ({module.weight.shape[0]} x {module.weight.shape[1]})")
                n_lines += 1
    print(f"\n  Total Linear params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\n  All trainable params: "
          f"{sum(p.numel() for p in model.parameters()):,} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.2f}M)")
"""
MultiConformerUniMol — wrapper around UniMolModel for K-conformer training.

Architecture (Pool AFTER head, TTA-style):

    inputs: (B, K, max_atoms, ...) — output of multi_conf_dataset collate_fn
        ↓ flatten to (B*K, max_atoms, ...)
    UniMolModel.forward (single pass):
        encoder → cls_repr (B*K, 512)
        classification_head → logits_per_conformer (B*K, output_dim)
        ↓ reshape to (B, K, output_dim)
        ↓ mean over K dim
    output: (B, output_dim)

Why pool AFTER head:
    Each conformer makes its own prediction → average predictions = TTA.
    Mathematically: y_pred = mean_k head(encoder(conf_k))
    
    Pros:
        - Każdy conformer gets full classification head capacity
        - Simple to implement (just reshape + mean after head)
        - Standard test-time augmentation pattern
    Cons (vs pool before head):
        - Less expressive: can't learn nonlinear K-conformer interactions

Inference for cache (Phase 1.5 internal, B2 strategy):
    Same wrapper, set return_repr=True path:
        encoder → cls_repr (B*K, 512)
        ↓ reshape to (B, K, 512)
        return as-is (no pooling, no head)
    → Phase 2 GP cache uses (N_mol, K, 512) tensor.

Init from pretrained:
    UniMolModel constructor auto-loads mol_pre_no_h_220816.pt.
    → No special handling needed; warm-started encoder + random-init head.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Output_dim resolution per task type
# ──────────────────────────────────────────────────────────────────────

def resolve_output_dim(task_type: str, num_tasks: int = 1) -> int:
    """
    Determine UniMolModel.output_dim for each task type.

    Convention matches unimol_tools default behavior:
        regression               → 1
        classification (binary)  → 2  (CrossEntropy on 2 logits)
        multilabel_classification→ num_tasks  (BCEWithLogits per task)
    """
    if task_type == "regression":
        return 1
    elif task_type == "classification":
        # UniMol convention: 2 logits even for binary, use CrossEntropy
        return 2
    elif task_type == "multilabel_classification":
        if num_tasks < 2:
            logger.warning(
                f"multilabel_classification with num_tasks={num_tasks} is unusual"
            )
        return num_tasks
    elif task_type == "multilabel_regression":
        return num_tasks
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# ──────────────────────────────────────────────────────────────────────
# Wrapper module
# ──────────────────────────────────────────────────────────────────────

class MultiConformerUniMol(nn.Module):
    """
    Wraps a UniMolModel to handle K conformers per molecule with mean pooling
    AFTER the classification head (TTA-style).

    Forward inputs: (B, K, max_atoms, ...)
    Forward outputs:
        - return_repr=False (default): logits (B, output_dim)
        - return_repr=True:            cls_repr (B, K, 512)  — for cache extraction

    Args:
        task_type:  one of {regression, classification, multilabel_classification,
                            multilabel_regression}
        num_tasks:  for multilabel cases, number of binary tasks
        remove_hs:  match what's expected by ConformerGen output (default True)
    """

    def __init__(
        self,
        task_type: str,
        num_tasks: int = 1,
        remove_hs: bool = True,
    ):
        super().__init__()
        from unimol_tools.models.unimol import UniMolModel

        self.task_type = task_type
        self.num_tasks = num_tasks
        self.output_dim = resolve_output_dim(task_type, num_tasks)
        self.remove_hs = remove_hs

        # UniMolModel constructor auto-loads mol_pre_no_h_220816.pt for warm start
        self.base = UniMolModel(
            output_dim=self.output_dim,
            data_type="molecule",
            remove_hs=remove_hs,
        )

    # ─────────────────────────────────────────────────────────────
    # Helpers: flatten/unflatten K dim
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_K(x: torch.Tensor) -> torch.Tensor:
        """(B, K, ...) → (B*K, ...) — collapse first 2 dims."""
        B, K = x.shape[:2]
        return x.reshape(B * K, *x.shape[2:])

    @staticmethod
    def _unflatten_K(x: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """(B*K, ...) → (B, K, ...) — restore K dim."""
        return x.reshape(B, K, *x.shape[1:])

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def forward(
        self,
        src_tokens: torch.Tensor,       # (B, K, L)
        src_distance: torch.Tensor,     # (B, K, L, L)
        src_coord: torch.Tensor,        # (B, K, L, 3)
        src_edge_type: torch.Tensor,    # (B, K, L, L)
        return_repr: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        return_repr=False (training):
            Pool AFTER head (TTA-style).
            Returns: logits (B, output_dim)

        return_repr=True (caching):
            Returns CLS embeddings without head/pooling.
            Returns: cls_repr (B, K, 512)
        """
        B, K = src_tokens.shape[:2]

        # Flatten K dim into batch
        flat_inputs = {
            "src_tokens":    self._flatten_K(src_tokens),     # (B*K, L)
            "src_distance":  self._flatten_K(src_distance),   # (B*K, L, L)
            "src_coord":     self._flatten_K(src_coord),      # (B*K, L, 3)
            "src_edge_type": self._flatten_K(src_edge_type),  # (B*K, L, L)
        }

        if return_repr:
            # Cache path: get CLS only, skip head, no pool
            out = self.base(**flat_inputs, return_repr=True)
            cls_repr = out["cls_repr"]                        # (B*K, 512)
            return self._unflatten_K(cls_repr, B, K)          # (B, K, 512)

        # Training path: full forward through head
        # UniMolModel.forward returns logits when return_repr=False
        logits_flat = self.base(**flat_inputs)                # (B*K, output_dim)

        # Reshape and pool AFTER head
        logits = self._unflatten_K(logits_flat, B, K)         # (B, K, output_dim)
        return logits.mean(dim=1)                             # (B, output_dim)

    # ─────────────────────────────────────────────────────────────
    # Convenience: forward in eval mode for cache extraction
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_cls(
        self,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_coord: torch.Tensor,
        src_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for cache extraction.
        Wrapper around forward(return_repr=True) with eval mode + no_grad.

        Returns: cls_repr (B, K, 512) on input device.
        """
        was_training = self.training
        self.eval()
        try:
            cls = self.forward(
                src_tokens, src_distance, src_coord, src_edge_type,
                return_repr=True,
            )
        finally:
            if was_training:
                self.train()
        return cls

    # ─────────────────────────────────────────────────────────────
    # Save/load — only base UniMolModel state
    # ─────────────────────────────────────────────────────────────

    def state_dict_base_only(self) -> Dict[str, torch.Tensor]:
        """
        Returns state_dict of self.base UniMolModel.
        Use this when saving phase1 weights → can be reloaded directly into
        a fresh UniMolModel for inference (without wrapper).
        """
        return self.base.state_dict()

    def load_state_dict_base_only(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ):
        """Load only the base UniMolModel weights (skip wrapper-specific buffers)."""
        return self.base.load_state_dict(state_dict, strict=strict)
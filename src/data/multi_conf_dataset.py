"""
Multi-conformer Dataset + collate_fns for Phase 1 training.

Workflow:
    1. ConformerGen sinh up to K conformers per molecule (boundary preserved).
    2. MolKConfDataset stores List[List[conf_dict]] + targets + valid_counts.
    3. collate_K_random_dup: train collate, sample K with replacement (epoch seeded).
    4. collate_K_fixed: valid/test collate, sample K with seed=0 (deterministic).
    5. Returned batch dict matches UniMol forward signature, but with extra K dim:
       (B, K, max_atoms, ...) → flatten to (B*K, ...) inside model wrapper.

Random duplication (user spec):
    Mỗi molecule sample K indices với replacement từ [0, valid_count_i)
    Vd K=5, valid_count=3:
       batch 1: indices [0,0,1,1,2] (= confs 1,1,2,2,3)
       batch 2: indices [0,1,2,2,2] (= confs 1,2,3,3,3)
       batch 3: indices [0,1,1,2,2] (= confs 1,2,2,3,3)
    → đúng spec "11223, 12333, 12233"

Reproducibility:
    Train collate uses torch.Generator seeded per-epoch by trainer.
    Valid/test collate uses Generator with manual_seed(fixed_seed) for deterministic.

Padding:
    Variable atom counts per conformer → pad to max_atoms in (B*K) flatten group.
    Padding token id = 0 (matches UniMolModel padding_idx default).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class MolKConfDataset(Dataset):
    """
    Dataset trả về (List[conf_dict], target) per molecule.

    conformers_per_mol: List[List[conf_dict]]
        Outer = molecule, inner = conformer dicts (variable length per molecule).
    targets: np.ndarray
        Shape (N_mol,) for regression/binary classification scalar targets.
        Shape (N_mol, n_tasks) for multilabel.
    """

    def __init__(
        self,
        conformers_per_mol: List[List[dict]],
        targets: np.ndarray,
    ):
        assert len(conformers_per_mol) == len(targets), \
            f"Length mismatch: confs={len(conformers_per_mol)}, targets={len(targets)}"
        # Filter molecules with 0 conformers (RDKit fail)
        keep = [i for i, c in enumerate(conformers_per_mol) if len(c) > 0]
        if len(keep) < len(conformers_per_mol):
            removed = len(conformers_per_mol) - len(keep)
            logger.warning(
                f"MolKConfDataset: removing {removed} molecules with 0 conformers"
            )
        self.conformers_per_mol = [conformers_per_mol[i] for i in keep]
        self.targets = targets[keep] if isinstance(targets, np.ndarray) else \
                       np.asarray([targets[i] for i in keep])
        self.kept_indices = np.array(keep, dtype=np.int64)
        self.valid_counts = np.array(
            [len(c) for c in self.conformers_per_mol], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.conformers_per_mol)

    def __getitem__(self, idx: int) -> Tuple[List[dict], np.ndarray]:
        return self.conformers_per_mol[idx], self.targets[idx]


# ──────────────────────────────────────────────────────────────────────
# Padding helpers
# ──────────────────────────────────────────────────────────────────────

def _pad_1d(a: np.ndarray, L: int, fill: int = 0) -> np.ndarray:
    o = np.full(L, fill, dtype=a.dtype)
    o[: len(a)] = a
    return o


def _pad_2d(a: np.ndarray, L: int, fill: int = 0) -> np.ndarray:
    o = np.full((L, L), fill, dtype=a.dtype)
    o[: a.shape[0], : a.shape[1]] = a
    return o


def _pad_coord(a: np.ndarray, L: int) -> np.ndarray:
    o = np.zeros((L, 3), dtype=a.dtype)
    o[: a.shape[0]] = a
    return o


# ──────────────────────────────────────────────────────────────────────
# Collate functions
# ──────────────────────────────────────────────────────────────────────

def _sample_K_indices(
    valid_count: int,
    K: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Random sample K indices in [0, valid_count) with replacement.

    Trường hợp valid_count >= K: vẫn sample with replacement → có thể trùng lặp,
    đảm bảo augmentation stochastic. Nếu muốn unique-when-possible, có thể switch
    sang torch.randperm — nhưng spec của user là replacement.
    """
    # randint: [low, high) with size
    return torch.randint(0, valid_count, (K,), generator=generator)


def _stack_K_conformers(
    conformers: List[dict],
    indices: torch.Tensor,
) -> List[dict]:
    """Pick K conformers from list at given indices (Python list indexing)."""
    return [conformers[int(i)] for i in indices]


def _collate_batch_with_indices(
    batch: List[Tuple[List[dict], np.ndarray]],
    K_indices_per_mol: List[torch.Tensor],
    K: int,
    target_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Stack batch of B molecules × K sampled conformers into batched tensors.

    Returns dict:
        src_tokens:    (B, K, L)
        src_distance:  (B, K, L, L)
        src_coord:     (B, K, L, 3)
        src_edge_type: (B, K, L, L)
        targets:       (B,) scalar or (B, n_tasks) for multilabel
    """
    B = len(batch)
    assert len(K_indices_per_mol) == B

    # First, gather all (B*K) conformer dicts to determine global max_atoms
    flat_confs: List[dict] = []
    for (confs, _), indices in zip(batch, K_indices_per_mol):
        flat_confs.extend(_stack_K_conformers(confs, indices))
    assert len(flat_confs) == B * K

    max_atoms = max(c["src_tokens"].shape[0] for c in flat_confs)

    # Pad each conformer to max_atoms, stack into (B*K, ...)
    src_tokens = np.stack([_pad_1d(c["src_tokens"], max_atoms) for c in flat_confs])
    src_distance = np.stack([_pad_2d(c["src_distance"], max_atoms) for c in flat_confs])
    src_coord = np.stack([_pad_coord(c["src_coord"], max_atoms) for c in flat_confs])
    src_edge_type = np.stack([_pad_2d(c["src_edge_type"], max_atoms) for c in flat_confs])

    # Reshape to (B, K, ...)
    out = {
        "src_tokens":    torch.from_numpy(src_tokens).reshape(B, K, max_atoms),
        "src_distance":  torch.from_numpy(src_distance).reshape(B, K, max_atoms, max_atoms),
        "src_coord":     torch.from_numpy(src_coord).reshape(B, K, max_atoms, 3),
        "src_edge_type": torch.from_numpy(src_edge_type).reshape(B, K, max_atoms, max_atoms),
    }

    # Targets
    targets = np.stack([t for _, t in batch])
    if targets.ndim == 1:
        out["targets"] = torch.tensor(targets, dtype=target_dtype)
    else:
        out["targets"] = torch.from_numpy(targets).to(target_dtype)

    return out


def make_collate_K_random_dup(K: int, generator: torch.Generator):
    """
    Factory: returns collate_fn that randomly samples K confs with replacement
    using the provided generator. Generator state advances per-call → stochastic
    augmentation across batches.

    Usage:
        gen = torch.Generator()
        gen.manual_seed(epoch_seed)
        loader = DataLoader(dataset, batch_size=B, collate_fn=make_collate_K_random_dup(K, gen))
    """
    def _collate(batch):
        K_indices_per_mol = [
            _sample_K_indices(len(confs), K, generator)
            for (confs, _) in batch
        ]
        return _collate_batch_with_indices(batch, K_indices_per_mol, K)

    return _collate


def make_collate_K_fixed(K: int, fixed_seed: int = 0):
    """
    Factory: returns collate_fn that uses deterministic sampling via fresh generator
    seeded by molecule-stable hash. Each molecule gets the same K indices regardless
    of batch composition.

    Implementation: per-molecule generator seeded by `fixed_seed * (idx_in_batch + 1)`
    inside the collate. To make sampling stable PER MOLECULE (independent of batch
    order), we need molecule-level seeding instead. Strategy:
      - Use molecule's valid_count + a stable index. But Dataset.__getitem__ doesn't
        pass molecule's global index by default.
      - Solution: use IndexedMolKConfDataset (subclass) that returns idx alongside data.

    Simpler alternative: for valid/test, sample order doesn't matter functionally
    because we're computing per-molecule predictions. Use ONE shared generator
    seeded by fixed_seed once → all molecules across all batches use deterministic
    sequence. This is sufficient for reproducibility.
    """
    def _collate(batch):
        # Fresh generator seeded each time → deterministic across calls
        gen = torch.Generator()
        gen.manual_seed(fixed_seed)
        K_indices_per_mol = [
            _sample_K_indices(len(confs), K, gen)
            for (confs, _) in batch
        ]
        return _collate_batch_with_indices(batch, K_indices_per_mol, K)

    return _collate


# ──────────────────────────────────────────────────────────────────────
# Indexed variant for true molecule-level deterministic sampling
# ──────────────────────────────────────────────────────────────────────

class IndexedMolKConfDataset(MolKConfDataset):
    """
    Same as MolKConfDataset but __getitem__ also returns molecule global index.
    Used for valid/test where we want sampling to be a function of (molecule, K, seed)
    independent of batch order.
    """
    def __getitem__(self, idx: int) -> Tuple[int, List[dict], np.ndarray]:
        return idx, self.conformers_per_mol[idx], self.targets[idx]


def make_collate_K_indexed_fixed(K: int, fixed_seed: int = 0):
    """
    Collate for IndexedMolKConfDataset: uses molecule index + fixed_seed to seed
    per-molecule generator → SAME conformer indices every call regardless of
    batch ordering or shuffle state.

    Use for valid/test deterministic forward.
    """
    def _collate(batch):
        # batch = [(idx, confs, target)]
        K_indices_per_mol = []
        for (mol_idx, confs, _) in batch:
            gen = torch.Generator()
            # Hash idx + seed deterministically
            gen.manual_seed(int(fixed_seed) * 1_000_003 + int(mol_idx))
            K_indices_per_mol.append(_sample_K_indices(len(confs), K, gen))

        # Reconstruct batch in (confs, target) format for shared collate
        plain_batch = [(confs, target) for (_, confs, target) in batch]
        out = _collate_batch_with_indices(plain_batch, K_indices_per_mol, K)
        # Also expose original molecule indices for trace-back
        out["mol_indices"] = torch.tensor([b[0] for b in batch], dtype=torch.long)
        return out

    return _collate
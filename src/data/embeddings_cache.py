"""
Embeddings cache for Step 3 (Phase 2 GP input).

Cache format:
    {
        'train': {'X':(N_tr, K, 512), 'valid_counts':(N_tr,) int, 'y':(N_tr,...), 'kept_indices':(N_tr,) int},
        'valid': {...},
        'test':  {...},
        'X_mean': (512,),
        'X_std':  (512,),
        'metadata': {dataset, split_seed, K_target, phase1_model_path, timestamp, D, remove_hs}
    }

Source of cache:
    Phase 1 trainer extracts cache directly via _extract_cls_cache().
    No more step1 weights dependency — UniMol is trained inside step 3 itself.

Normalize:
    X_mean, X_std computed from TRAIN valid conformers (padding slots ignored).
    Applied to all 3 splits before saving.

Sampling helpers:
    sample_K_with_duplication: per-gen random dup for phase 2 train
    fixed_K_sample:            deterministic sampling for phase 2 valid/test
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────

def get_cache_path(cache_dir: str, dataset: str, split_seed: int, K: int) -> str:
    """
    Standardize cache path.
    Example: data/embeddings_cache/esol/seed_0/K_10.pt
    """
    return os.path.join(cache_dir, dataset, f"seed_{split_seed}", f"K_{K}.pt")


def cache_exists(cache_dir: str, dataset: str, split_seed: int, K: int) -> bool:
    return os.path.exists(get_cache_path(cache_dir, dataset, split_seed, K))


# ──────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────

def save_cache(cache_path: str, cache_dict: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_dict, cache_path)
    logger.info(f"Cache saved: {cache_path}")


def load_cache(cache_path: str, map_location="cpu") -> dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return torch.load(cache_path, map_location=map_location, weights_only=False)


# ──────────────────────────────────────────────────────────────────────
# Normalize stats (train only) — applied to all splits
# ──────────────────────────────────────────────────────────────────────

def compute_normalize_stats(
    X: torch.Tensor,
    valid_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean/std on VALID conformers only (excluding padding slots).

    X:              (N_mol, K, D)
    valid_counts:   (N_mol,)
    Returns: mean (D,), std (D,)
    """
    N_mol, K, D = X.shape
    mask = torch.arange(K).unsqueeze(0) < valid_counts.unsqueeze(1)
    valid_X = X[mask]
    mean = valid_X.mean(dim=0)
    std = valid_X.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_normalize(
    X: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Apply (X - mean) / std. Padding slots become (-mean/std), but they're never sampled."""
    return (X - mean.view(1, 1, -1)) / std.view(1, 1, -1)


# ──────────────────────────────────────────────────────────────────────
# Assemble cache from phase 1 output
# ──────────────────────────────────────────────────────────────────────

def assemble_cache_dict(
    cache_train: Dict[str, torch.Tensor],
    cache_valid: Dict[str, torch.Tensor],
    cache_test: Dict[str, torch.Tensor],
    *,
    dataset: str,
    split_seed: int,
    K: int,
    phase1_model_path: str,
    phase1_timestamp: str,
    remove_hs: bool = True,
) -> dict:
    """
    Combine 3 raw split caches from Phase 1 into final cache dict format.

    Steps:
        1. Compute normalize stats from TRAIN valid conformers
        2. Apply normalize to all 3 splits
        3. Wrap with metadata
    """
    # Compute normalize from train
    X_mean, X_std = compute_normalize_stats(
        cache_train["X"], cache_train["valid_counts"]
    )
    logger.info(
        f"Normalize stats from train: "
        f"mean range [{X_mean.min():.4f}, {X_mean.max():.4f}], "
        f"std range [{X_std.min():.4f}, {X_std.max():.4f}]"
    )

    # Apply to all 3 splits
    cache_train["X"] = apply_normalize(cache_train["X"], X_mean, X_std)
    cache_valid["X"] = apply_normalize(cache_valid["X"], X_mean, X_std)
    cache_test["X"] = apply_normalize(cache_test["X"], X_mean, X_std)

    D = cache_train["X"].shape[-1]

    return {
        "train":  cache_train,
        "valid":  cache_valid,
        "test":   cache_test,
        "X_mean": X_mean,
        "X_std":  X_std,
        "metadata": {
            "dataset":           dataset,
            "split_seed":        split_seed,
            "K_target":          K,
            "phase1_model_path": phase1_model_path,
            "phase1_timestamp":  phase1_timestamp,
            "timestamp":         datetime.now().strftime("%Y%m%d_%H%M%S"),
            "D":                 D,
            "remove_hs":         remove_hs,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# PCA reduction (workaround evogp CUDA stack limit)
# ──────────────────────────────────────────────────────────────────────

def reduce_and_normalize_splits(
    cache: dict,
    gp_input_dim: int,
) -> Tuple[dict, dict]:
    """
    Apply PCA reduction + standardize for GP input.

    Evogp CUDA kernel has a hard limit on number of variables referenced in tree
    (MAX_STACK). With 512-dim UniMol CLS, assertion `varLen <= MAX_STACK / 4` fails
    → must reduce.

    Pipeline:
        1. Extract train valid conformers (flattened, N_conformers × D)
        2. Fit PCA on train → components (D, gp_input_dim), pca_mean (D,)
        3. Apply PCA to all 3 splits
        4. Fit standardize stats (mean/std) in reduced space on train
        5. Apply standardize to all 3 splits

    Args:
        cache:         dict from load_cache() — X already normalized in 512-space
        gp_input_dim:  target dim (e.g. 64 or 128)

    Returns:
        reduced_cache: same structure, but X shape (N, K, gp_input_dim)
        transform_info: dict with PCA components + normalize stats
    """
    from sklearn.decomposition import PCA

    D_orig = cache["train"]["X"].shape[-1]
    K = cache["train"]["X"].shape[1]

    if gp_input_dim > D_orig:
        raise ValueError(f"gp_input_dim ({gp_input_dim}) > D_orig ({D_orig})")
    if gp_input_dim == D_orig:
        logger.warning(
            f"gp_input_dim={gp_input_dim} equals D_orig={D_orig}. "
            f"Skipping PCA; applying standardize only."
        )

    # ── Step 1: extract train valid conformers ──
    X_train = cache["train"]["X"]
    vc_train = cache["train"]["valid_counts"]
    mask = torch.arange(K).unsqueeze(0) < vc_train.unsqueeze(1)
    train_flat = X_train[mask].numpy()
    logger.info(
        f"PCA fit on {train_flat.shape[0]} train conformers, "
        f"{D_orig} → {gp_input_dim} dim"
    )

    # ── Step 2: fit PCA ──
    if gp_input_dim < D_orig:
        pca = PCA(n_components=gp_input_dim)
        pca.fit(train_flat)
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_var:.2%}")
        components = torch.from_numpy(pca.components_.T).float()
        pca_mean = torch.from_numpy(pca.mean_).float()
    else:
        components = torch.eye(D_orig, dtype=torch.float32)
        pca_mean = torch.zeros(D_orig, dtype=torch.float32)
        explained_var = 1.0

    # ── Step 3: apply PCA to all 3 splits ──
    def _apply_pca(X: torch.Tensor) -> torch.Tensor:
        N, K_, D = X.shape
        X_flat = X.reshape(-1, D)
        X_reduced_flat = (X_flat - pca_mean) @ components
        return X_reduced_flat.reshape(N, K_, gp_input_dim)

    reduced = {}
    for split in ("train", "valid", "test"):
        reduced[split] = {
            "X":            _apply_pca(cache[split]["X"]),
            "valid_counts": cache[split]["valid_counts"],
            "y":            cache[split]["y"],
            "kept_indices": cache[split]["kept_indices"],
        }

    # ── Step 4-5: compute + apply standardize stats on reduced train ──
    X_tr_red = reduced["train"]["X"]
    mask = torch.arange(K).unsqueeze(0) < reduced["train"]["valid_counts"].unsqueeze(1)
    train_red_flat = X_tr_red[mask]
    reduced_mean = train_red_flat.mean(dim=0)
    reduced_std = train_red_flat.std(dim=0).clamp(min=1e-6)

    for split in ("train", "valid", "test"):
        reduced[split]["X"] = (reduced[split]["X"] - reduced_mean.view(1, 1, -1)) / reduced_std.view(1, 1, -1)

    # Preserve other cache fields
    reduced["X_mean"] = cache["X_mean"]
    reduced["X_std"] = cache["X_std"]
    reduced["metadata"] = {
        **cache["metadata"],
        "gp_input_dim":      gp_input_dim,
        "pca_explained_var": float(explained_var),
    }

    transform_info = {
        "pca_components":   components,
        "pca_mean":         pca_mean,
        "reduced_mean":     reduced_mean,
        "reduced_std":      reduced_std,
        "D_orig":           D_orig,
        "gp_input_dim":     gp_input_dim,
        "pca_explained_var": float(explained_var),
    }

    return reduced, transform_info


# ──────────────────────────────────────────────────────────────────────
# Sampling helpers (Phase 2 GP input)
# ──────────────────────────────────────────────────────────────────────

def sample_K_with_duplication(
    X: torch.Tensor,
    valid_counts: torch.Tensor,
    K: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    For each molecule, sample K indices with replacement from [0, valid_counts[i]).
    If valid_counts[i] < K: molecule will have duplicate conformers.
    If valid_counts[i] >= K: still sample with replacement (stochastic augmentation).
    """
    N_mol = X.shape[0]
    device = X.device

    u = torch.rand(N_mol, K, generator=generator, device=device)
    idx = (u * valid_counts.to(device).float().unsqueeze(1)).long()
    idx = idx.clamp(max=valid_counts.to(device).unsqueeze(1) - 1)

    mol_idx = torch.arange(N_mol, device=device).unsqueeze(1).expand(-1, K)
    return X[mol_idx, idx]


def fixed_K_sample(
    X: torch.Tensor,
    valid_counts: torch.Tensor,
    K: int,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic one-time sampling for valid/test using seeded Generator."""
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    return sample_K_with_duplication(X, valid_counts, K, generator=g)
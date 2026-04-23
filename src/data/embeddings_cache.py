"""
Embeddings cache for Step 3.

Vai trò:
    - Load UniMol từ step 1 weights (frozen).
    - Sinh K conformers per molecule (RDKit), forward qua UniMol encoder,
      lấy CLS token (512-dim per conformer).
    - Lưu cache `.pt` per (dataset, split_seed, K), chứa 3 splits train/valid/test.
    - Cung cấp hàm `sample_K_with_duplication` để resample K conformers
      với random duplication từ K_i conformers thực sinh được.

Cache format:
    {
        'train': {'X':(N_tr, K, 512), 'valid_counts':(N_tr,) int, 'y':(N_tr,), 'kept_indices':(N_tr,) int},
        'valid': {...},
        'test':  {...},
        'X_mean': (512,),          # từ train valid conformers
        'X_std':  (512,),
        'metadata': {dataset, split_seed, K_target, step1_weight_path, timestamp, D, remove_hs}
    }

Normalize:
    X_mean, X_std tính từ TRAIN valid conformers (padding slots không kể).
    Apply cho cả 3 splits trước khi lưu → tensor X đã normalized.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

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


def find_step1_weights(
    dataset: str,
    split_seed: int,
    output_dir: str = "experiments",
    explicit_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Find step 1 model_0.pth.

    - Nếu `explicit_path` truyền vào và tồn tại → dùng.
    - Nếu không, tìm trong experiments/step1/{dataset}/seed_{s}/*/model_0.pth,
      chọn timestamp mới nhất.

    Returns:
        (weight_path, timestamp_str)
    """
    if explicit_path is not None:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Step 1 weights not found: {explicit_path}")
        ts = os.path.basename(os.path.dirname(explicit_path))
        return explicit_path, ts

    step1_dir = os.path.join(output_dir, "step1", dataset, f"seed_{split_seed}")
    if not os.path.isdir(step1_dir):
        raise FileNotFoundError(
            f"No step 1 runs found at {step1_dir}. "
            f"Chạy run_step1.py cho {dataset} seed_{split_seed} trước."
        )
    timestamps = sorted(
        d for d in os.listdir(step1_dir)
        if os.path.isdir(os.path.join(step1_dir, d))
    )
    if not timestamps:
        raise FileNotFoundError(f"No timestamp directories in {step1_dir}")
    latest = timestamps[-1]
    weight_path = os.path.join(step1_dir, latest, "model_0.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"model_0.pth not found in latest step 1 run {latest} at {weight_path}"
        )
    return weight_path, latest


# ──────────────────────────────────────────────────────────────────────
# Load frozen UniMol model
# ──────────────────────────────────────────────────────────────────────

def load_step1_unimol(weight_path: str, device: torch.device):
    """
    Load UniMolModel (unimolv1) with step 1 trained weights, set frozen.

    Note: strict=False vì state_dict từ step 1 có thêm classification_head,
    nhưng chỉ encoder + gbf + gbf_proj được dùng (return_repr=True).
    """
    from unimol_tools.models.unimol import UniMolModel

    model = UniMolModel(output_dim=1, data_type="molecule", remove_hs=True)

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading step 1 weights: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys when loading step 1 weights: {unexpected[:5]}...")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ──────────────────────────────────────────────────────────────────────
# Conformer generation (per-molecule, preserve boundaries)
# ──────────────────────────────────────────────────────────────────────

def _generate_conformers_per_molecule(
    smiles_list: List[str],
    K: int,
    seed: int = 42,
    remove_hs: bool = True,
) -> List[List[dict]]:
    """
    Sinh đến K conformers per molecule, return per-molecule list.
    KHÔNG flatten → giữ nguyên boundary để biết K_i per molecule.

    Failed conformer = src_coord toàn 0 → loại khỏi list của molecule đó.
    """
    from unimol_tools.data.conformer import ConformerGen

    # ConformerGen.single_process(smiles) trả về List[feat_dict] với len = K (hoặc ít hơn nếu fail).
    cg = ConformerGen(
        data_type="molecule",
        remove_hs=remove_hs,
        n_confomer=K,
        seed=seed,
        multi_process=False,  # Tắt multi_process để dễ debug + giữ ordering
    )

    conformers_per_mol: List[List[dict]] = []
    for smi in tqdm(smiles_list, desc=f"Generating K={K} conformers"):
        try:
            feats = cg.single_process(smi)  # List[feat_dict], len up to K
        except Exception as e:
            logger.warning(f"ConformerGen failed for SMILES '{smi}': {e}")
            feats = []

        # Filter failed conformers (src_coord all zero means 3D gen failed)
        valid_feats = [
            f for f in feats
            if not (f["src_coord"] == 0.0).all()
        ]
        conformers_per_mol.append(valid_feats)

    return conformers_per_mol


# ──────────────────────────────────────────────────────────────────────
# Batched UniMol forward → CLS embeddings
# ──────────────────────────────────────────────────────────────────────

def _pad_and_batch(conf_list: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Pad 1 batch of conformers (variable atom count) thành tensor batched.
    Trả về dict với src_tokens/src_distance/src_coord/src_edge_type trên device.
    """
    max_len = max(c["src_tokens"].shape[0] for c in conf_list)

    def pad1d(a, L, v=0):
        o = np.full(L, v, dtype=a.dtype)
        o[: len(a)] = a
        return o

    def pad2d(a, L, v=0):
        o = np.full((L, L), v, dtype=a.dtype)
        o[: a.shape[0], : a.shape[1]] = a
        return o

    def padcoord(a, L):
        o = np.zeros((L, 3), dtype=a.dtype)
        o[: a.shape[0]] = a
        return o

    return {
        "src_tokens": torch.tensor(
            np.stack([pad1d(c["src_tokens"], max_len) for c in conf_list])
        ).to(device),
        "src_distance": torch.tensor(
            np.stack([pad2d(c["src_distance"], max_len) for c in conf_list])
        ).to(device),
        "src_coord": torch.tensor(
            np.stack([padcoord(c["src_coord"], max_len) for c in conf_list])
        ).to(device),
        "src_edge_type": torch.tensor(
            np.stack([pad2d(c["src_edge_type"], max_len) for c in conf_list])
        ).to(device),
    }


def _forward_cls_embeddings(
    flat_confs: List[dict],
    model,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Forward tất cả conformers qua UniMol (frozen, no_grad), lấy CLS per conformer.
    Returns: (total_confs, 512) float32 CPU tensor.
    """
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(flat_confs), batch_size), desc="UniMol forward"):
            batch_confs = flat_confs[i : i + batch_size]
            batch = _pad_and_batch(batch_confs, device)
            out = model(**batch, return_repr=True)
            cls = out["cls_repr"]  # (B, 512)
            embeddings.append(cls.detach().cpu().float())
    return torch.cat(embeddings, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Main compute: orchestrate conformers → embeddings → (N_mol, K, D) tensor
# ──────────────────────────────────────────────────────────────────────

def compute_embeddings_for_split(
    smiles_list: List[str],
    y_list,
    unimol_model,
    K: int,
    device: torch.device,
    batch_size: int = 64,
    seed: int = 42,
    remove_hs: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute padded embeddings cho 1 split.

    Returns dict:
        X:              (N_kept, K, 512) float32 — CLS embeddings, padded zeros nếu K_i < K
        valid_counts:   (N_kept,) int — K_i per molecule (>= 1, molecules với K_i=0 đã loại)
        y:              (N_kept,) float32 — target values (đã filter cho valid molecules)
        kept_indices:   (N_kept,) int — indices trong smiles_list gốc được giữ lại

    Molecules với 0 conformers (RDKit fail hoàn toàn) được LOẠI, log warning.
    """
    assert K >= 1, "K must be >= 1"

    # Step A: generate conformers, boundary-preserving
    conformers_per_mol = _generate_conformers_per_molecule(
        smiles_list, K=K, seed=seed, remove_hs=remove_hs
    )

    # Filter molecules with 0 valid conformers
    kept_indices = [i for i, confs in enumerate(conformers_per_mol) if len(confs) > 0]
    removed = len(smiles_list) - len(kept_indices)
    if removed > 0:
        logger.warning(
            f"Removed {removed}/{len(smiles_list)} molecules with 0 valid conformers "
            f"(RDKit fail). Kept {len(kept_indices)}."
        )

    conformers_per_mol = [conformers_per_mol[i] for i in kept_indices]
    y_filtered = [y_list[i] for i in kept_indices]
    N_mol = len(conformers_per_mol)

    if N_mol == 0:
        raise RuntimeError("All molecules failed conformer generation!")

    valid_counts = torch.tensor(
        [min(len(c), K) for c in conformers_per_mol], dtype=torch.long
    )
    # Trim to K (nếu RDKit sinh nhiều hơn K — hiếm, nhưng phòng)
    conformers_per_mol = [c[:K] for c in conformers_per_mol]

    # Step B: flatten for batched UniMol forward
    flat_confs: List[dict] = []
    flat_mol_idx: List[int] = []
    flat_conf_idx: List[int] = []
    for mol_i, confs in enumerate(conformers_per_mol):
        for conf_i, conf in enumerate(confs):
            flat_confs.append(conf)
            flat_mol_idx.append(mol_i)
            flat_conf_idx.append(conf_i)

    logger.info(
        f"Forwarding {len(flat_confs)} conformers from {N_mol} molecules "
        f"(avg {len(flat_confs)/N_mol:.2f} conformers/mol)"
    )

    # Step C: batched UniMol forward
    flat_embeddings = _forward_cls_embeddings(
        flat_confs, unimol_model, device, batch_size=batch_size
    )  # (total_confs, 512)

    D = flat_embeddings.shape[1]

    # Step D: scatter back to (N_mol, K, D)
    X = torch.zeros(N_mol, K, D, dtype=torch.float32)
    for flat_i, (mol_i, conf_i) in enumerate(zip(flat_mol_idx, flat_conf_idx)):
        X[mol_i, conf_i] = flat_embeddings[flat_i]

    y_tensor = torch.tensor(np.asarray(y_filtered, dtype=np.float32), dtype=torch.float32)
    if y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
        y_tensor = y_tensor.squeeze(1)

    return {
        "X": X,
        "valid_counts": valid_counts,
        "y": y_tensor,
        "kept_indices": torch.tensor(kept_indices, dtype=torch.long),
    }


# ──────────────────────────────────────────────────────────────────────
# Normalize stats (train only) — applied to all splits
# ──────────────────────────────────────────────────────────────────────

def compute_normalize_stats(X: torch.Tensor, valid_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean/std trên CHỈ valid conformers (không tính padding zeros).

    X:              (N_mol, K, D)
    valid_counts:   (N_mol,)
    Returns: mean (D,), std (D,)
    """
    N_mol, K, D = X.shape
    # Build a (N_mol, K) boolean mask: True nếu conformer slot là valid
    mask = torch.arange(K).unsqueeze(0) < valid_counts.unsqueeze(1)  # (N_mol, K)
    valid_X = X[mask]  # (total_valid_confs, D)
    mean = valid_X.mean(dim=0)  # (D,)
    std = valid_X.std(dim=0).clamp(min=1e-6)  # (D,) — avoid div by 0
    return mean, std


def apply_normalize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Apply (X - mean) / std. Padding slots (all zeros) sẽ thành (-mean/std) — không
    matter vì ta không sample chúng (valid_counts giới hạn).
    """
    return (X - mean.view(1, 1, -1)) / std.view(1, 1, -1)


# ──────────────────────────────────────────────────────────────────────
# Save / Load / full pipeline
# ──────────────────────────────────────────────────────────────────────

def save_cache(cache_path: str, cache_dict: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_dict, cache_path)
    logger.info(f"Cache saved: {cache_path}")


def load_cache(cache_path: str, map_location="cpu") -> dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return torch.load(cache_path, map_location=map_location, weights_only=False)


def build_and_save_cache(
    train_df,
    valid_df,
    test_df,
    *,
    dataset: str,
    split_seed: int,
    K: int,
    smiles_column: str,
    target_column: str,
    cache_dir: str,
    device: torch.device,
    step1_weight_path: str,
    step1_timestamp: str = "",
    batch_size: int = 64,
    conf_seed: int = 42,
    remove_hs: bool = True,
) -> str:
    """
    Full pipeline: precompute cache cho cả 3 splits, normalize, save.

    Returns: cache_path
    """
    # Load frozen UniMol
    logger.info(f"Loading UniMol with step 1 weights from {step1_weight_path}")
    model = load_step1_unimol(step1_weight_path, device)

    # Compute each split
    splits = {}
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        logger.info(f"\n=== Computing embeddings for '{split_name}' ({len(df)} molecules) ===")
        splits[split_name] = compute_embeddings_for_split(
            smiles_list=df[smiles_column].tolist(),
            y_list=df[target_column].values,
            unimol_model=model,
            K=K,
            device=device,
            batch_size=batch_size,
            seed=conf_seed,
            remove_hs=remove_hs,
        )

    # Compute normalize stats từ train only
    logger.info("Computing normalize stats from train...")
    X_mean, X_std = compute_normalize_stats(
        splits["train"]["X"], splits["train"]["valid_counts"]
    )

    # Apply normalize to all 3 splits
    for split_name in splits:
        splits[split_name]["X"] = apply_normalize(
            splits[split_name]["X"], X_mean, X_std
        )

    # Assemble cache
    cache_dict = {
        "train": splits["train"],
        "valid": splits["valid"],
        "test": splits["test"],
        "X_mean": X_mean,
        "X_std": X_std,
        "metadata": {
            "dataset": dataset,
            "split_seed": split_seed,
            "K_target": K,
            "step1_weight_path": step1_weight_path,
            "step1_timestamp": step1_timestamp,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "D": splits["train"]["X"].shape[-1],
            "remove_hs": remove_hs,
            "conf_seed": conf_seed,
        },
    }

    cache_path = get_cache_path(cache_dir, dataset, split_seed, K)
    save_cache(cache_path, cache_dict)

    # Log summary
    logger.info("\n=== Cache summary ===")
    for split_name in ("train", "valid", "test"):
        s = splits[split_name]
        vc = s["valid_counts"]
        logger.info(
            f"  {split_name}: N_mol={s['X'].shape[0]}, "
            f"valid_counts min/mean/max = {vc.min().item()}/{vc.float().mean().item():.2f}/{vc.max().item()}"
        )
    logger.info(f"  X_mean range: [{X_mean.min():.4f}, {X_mean.max():.4f}]")
    logger.info(f"  X_std  range: [{X_std.min():.4f}, {X_std.max():.4f}]")

    return cache_path


# ──────────────────────────────────────────────────────────────────────
# Sampling helper (train uses per-gen, valid/test uses fixed)
# ──────────────────────────────────────────────────────────────────────

def sample_K_with_duplication(
    X: torch.Tensor,
    valid_counts: torch.Tensor,
    K: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Với mỗi molecule, sample K indices với replacement từ [0, valid_counts[i]).
    Nếu valid_counts[i] < K: molecules sẽ có duplicate conformers.
    Nếu valid_counts[i] == K: sampling from [0,K) with replacement — vẫn có thể duplicate
                              (vì ta muốn stochastic augmentation, không chỉ permutation).

    Args:
        X:            (N_mol, K_cache, D) — cache tensor
        valid_counts: (N_mol,) int — K_i per molecule
        K:            target K (usually K_cache, nhưng có thể < nếu muốn sub-sample)
        generator:    optional torch Generator for reproducibility

    Returns:
        X_sampled:    (N_mol, K, D)
    """
    N_mol = X.shape[0]
    device = X.device

    # Random floats in [0, 1), scale by valid_counts, floor
    u = torch.rand(N_mol, K, generator=generator, device=device)
    idx = (u * valid_counts.to(device).float().unsqueeze(1)).long()  # (N_mol, K)
    # Clamp for safety (in case of floating-point issues at boundary)
    idx = idx.clamp(max=valid_counts.to(device).unsqueeze(1) - 1)

    # Advanced indexing
    mol_idx = torch.arange(N_mol, device=device).unsqueeze(1).expand(-1, K)
    return X[mol_idx, idx]  # (N_mol, K, D)


def fixed_K_sample(
    X: torch.Tensor,
    valid_counts: torch.Tensor,
    K: int,
    seed: int = 0,
) -> torch.Tensor:
    """
    Deterministic one-time sampling for valid/test. Dùng seeded Generator
    để mỗi lần gọi với cùng (X, valid_counts, seed) đều trả về cùng kết quả.
    """
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    return sample_K_with_duplication(X, valid_counts, K, generator=g)


# ──────────────────────────────────────────────────────────────────────
# PCA reduction (workaround evogp CUDA stack limit)
# ──────────────────────────────────────────────────────────────────────

def reduce_and_normalize_splits(
    cache: dict,
    gp_input_dim: int,
) -> Tuple[dict, dict]:
    """
    Apply PCA reduction + standardize cho GP input.

    Evogp CUDA kernel có hard limit về số biến referenced trong tree (MAX_STACK).
    Với 512-dim UniMol CLS, assertion `varLen <= MAX_STACK / 4` fail → phải reduce.

    Pipeline:
        1. Extract train valid conformers (flattened, N_conformers × D)
        2. Fit PCA trên train → components (D, gp_input_dim), pca_mean (D,)
        3. Apply PCA cho cả 3 splits
        4. Fit standardize stats (mean/std) trong reduced space trên train
        5. Apply standardize cho cả 3 splits

    Args:
        cache:         dict từ load_cache() — X đã normalized ở 512-space
        gp_input_dim:  target dim (e.g. 64)

    Returns:
        reduced_cache: dict mới cùng structure như cache nhưng X shape (N, K, d_gp)
        transform_info: dict chứa PCA components + normalize stats (để save/reproduce)
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
    X_train = cache["train"]["X"]            # (N_tr, K, D)
    vc_train = cache["train"]["valid_counts"]
    mask = torch.arange(K).unsqueeze(0) < vc_train.unsqueeze(1)     # (N_tr, K)
    train_flat = X_train[mask].numpy()                              # (N_valid_conf, D)
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
        components = torch.from_numpy(pca.components_.T).float()    # (D_orig, gp_input_dim)
        pca_mean   = torch.from_numpy(pca.mean_).float()             # (D_orig,)
    else:
        # No reduction — identity transformation
        components = torch.eye(D_orig, dtype=torch.float32)
        pca_mean   = torch.zeros(D_orig, dtype=torch.float32)
        explained_var = 1.0

    # ── Step 3: apply PCA to all 3 splits ──
    def _apply_pca(X: torch.Tensor) -> torch.Tensor:
        N, K_, D = X.shape
        X_flat = X.reshape(-1, D)                                    # (N*K, D_orig)
        X_reduced_flat = (X_flat - pca_mean) @ components            # (N*K, gp_input_dim)
        return X_reduced_flat.reshape(N, K_, gp_input_dim)

    reduced = {}
    for split in ("train", "valid", "test"):
        reduced[split] = {
            "X":             _apply_pca(cache[split]["X"]),
            "valid_counts":  cache[split]["valid_counts"],
            "y":             cache[split]["y"],
            "kept_indices":  cache[split]["kept_indices"],
        }

    # ── Step 4-5: compute + apply standardize stats on reduced train ──
    X_tr_red = reduced["train"]["X"]                                 # (N_tr, K, d_gp)
    mask = torch.arange(K).unsqueeze(0) < reduced["train"]["valid_counts"].unsqueeze(1)
    train_red_flat = X_tr_red[mask]                                  # (N_valid_conf, d_gp)
    reduced_mean = train_red_flat.mean(dim=0)
    reduced_std = train_red_flat.std(dim=0).clamp(min=1e-6)

    for split in ("train", "valid", "test"):
        reduced[split]["X"] = (reduced[split]["X"] - reduced_mean.view(1, 1, -1)) / reduced_std.view(1, 1, -1)

    # Preserve other cache fields
    reduced["X_mean"]   = cache["X_mean"]       # original 512-dim stats (info)
    reduced["X_std"]    = cache["X_std"]
    reduced["metadata"] = {**cache["metadata"], "gp_input_dim": gp_input_dim,
                            "pca_explained_var": float(explained_var)}

    transform_info = {
        "pca_components":   components,          # (D_orig, gp_input_dim)
        "pca_mean":         pca_mean,            # (D_orig,)
        "reduced_mean":     reduced_mean,        # (gp_input_dim,)
        "reduced_std":      reduced_std,         # (gp_input_dim,)
        "D_orig":           D_orig,
        "gp_input_dim":     gp_input_dim,
        "pca_explained_var": float(explained_var),
    }

    return reduced, transform_info
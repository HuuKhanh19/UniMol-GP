"""
00 - Extract Unimol-v1 embeddings for ESOL train / val splits.

Run once. Caches (X, y) for train and val as .pt files. Subsequent
experiments load these instead of re-running the (slow) Unimol forward.

Input:
    data/raw/refined_ESOL.csv  with columns SMILES, TARGET (and optionally VALID)

Output:
    data/cache/esol_train.pt   {'X': (N_train, 512), 'y': (N_train,)}
    data/cache/esol_val.pt     {'X': (N_val,   512), 'y': (N_val,)}
    data/cache/esol_meta.json  split sizes + checksum

Usage:
    python experiments/exp_A_sanity/00_extract_embeddings.py \\
        --csv data/raw/refined_ESOL.csv \\
        --out data/cache \\
        --split-frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_split(
    csv_path: Path,
    split_frac: float,
    seed: int,
    smiles_col: str | None = None,
    target_col: str | None = None,
):
    """Load ESOL CSV. Use VALID column if present (0=train, 1=val), else random split.

    Column auto-detection (case-insensitive):
        SMILES candidates: SMILES, smiles, smi, Smiles
        Target candidates: TARGET, target, measured, y, label,
                           'measured log solubility in mols per litre'

    Use --smiles-col / --target-col to override.
    """
    df = pd.read_csv(csv_path)
    print(f"  CSV columns: {list(df.columns)}")

    # ---- SMILES column ----
    if smiles_col is None:
        for cand in ["SMILES", "smiles", "smi", "Smiles", "SMI"]:
            if cand in df.columns:
                smiles_col = cand
                break
    if smiles_col is None or smiles_col not in df.columns:
        raise ValueError(
            f"Could not auto-detect SMILES column. Use --smiles-col. CSV columns: {list(df.columns)}"
        )

    # ---- Target column ----
    if target_col is None:
        # Order matters: most explicit first
        candidates = [
            "TARGET", "target",
            "measured log solubility in mols per litre",
            "measured", "Measured",
            "logS", "logSolubility",
            "y", "label", "Y",
        ]
        for cand in candidates:
            if cand in df.columns:
                target_col = cand
                break
    if target_col is None or target_col not in df.columns:
        raise ValueError(
            f"Could not auto-detect target column. Use --target-col. CSV columns: {list(df.columns)}"
        )

    print(f"  using smiles_col='{smiles_col}', target_col='{target_col}'")
    df = df.rename(columns={smiles_col: "SMILES", target_col: "TARGET"})

    if "VALID" in df.columns:
        train_df = df[df["VALID"] == 0].reset_index(drop=True)
        val_df = df[df["VALID"] == 1].reset_index(drop=True)
        print(f"  using VALID column from CSV: train={len(train_df)}, val={len(val_df)}")
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_val = int(len(df) * split_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        print(f"  random split (seed={seed}, val_frac={split_frac}): train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


def extract_embeddings(
    smiles_list,
    data_type: str = "molecule",
    remove_hs: bool = False,
    batch_size: int = 32,
    n_confomer: int = 1,
    seed: int = 42,
    device: str | None = None,
) -> np.ndarray:
    """Extract Unimol cls_repr embeddings DIRECTLY via UniMolModel + ConformerGen.

    Bypasses `UniMolRepr.get_repr()` because some forks have a broken MolDataset
    that doesn't auto-process SMILES strings into conformer dicts (resulting in
    'str' object has no attribute 'keys' errors during inference).

    Args:
        smiles_list: list of SMILES strings.
        n_confomer: k (number of conformers per molecule). If k>1, returns mean
            over conformers automatically.

    Returns:
        (N, 512) numpy array of cls_repr.
    """
    import torch
    from tqdm import tqdm

    # Import inside function so we only require unimol_tools at runtime
    from unimol_tools.models import UniMolModel
    from unimol_tools.data.conformer import ConformerGen

    dev = torch.device(device if device is not None
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device = {dev}")

    # 1. Load pre-trained Unimol model (weights auto-downloaded if needed)
    print(f"  loading UniMolModel (data_type={data_type}, remove_hs={remove_hs}) ...")
    model = UniMolModel(output_dim=2, data_type=data_type, remove_hs=remove_hs)
    model = model.to(dev).eval()

    # 2. Generate conformers via ConformerGen (the actual workhorse inside unimol_tools)
    conf_gen = ConformerGen(
        method="rdkit_random",
        mode="fast",
        n_confomer=n_confomer,
        remove_hs=remove_hs,
        dictionary=model.dictionary,
        max_atoms=256,
        multi_process=False,   # avoid Windows multiprocessing quirks
        seed=seed,
    )
    print(f"  generating conformers (k={n_confomer}) for {len(smiles_list)} molecules ...")
    inputs, _ = conf_gen.transform(smiles_list)
    print(f"  got {len(inputs)} conformer dicts "
          f"(expected {len(smiles_list) * n_confomer})")

    # 3. Batch forward through model with return_repr=True
    embeddings = []
    n_batches = (len(inputs) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(inputs), batch_size), total=n_batches, desc="forward Unimol"):
        batch_dicts = inputs[i : i + batch_size]
        # batch_collate_fn expects list of (sample_dict, label) — label is dummy here
        batch, _ = model.batch_collate_fn([(d, 0.0) for d in batch_dicts])
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch, return_repr=True, return_atomic_reprs=False)
        cls_repr = out["cls_repr"].detach().cpu().float().numpy()  # (B, 512)
        embeddings.append(cls_repr)
    embeddings = np.concatenate(embeddings, axis=0)  # (N*k, 512) or (N, 512)

    # 4. If k > 1, mean over conformers per molecule
    if n_confomer > 1:
        assert embeddings.shape[0] == len(smiles_list) * n_confomer, (
            f"shape mismatch: got {embeddings.shape[0]} embeddings for "
            f"{len(smiles_list)} mols x {n_confomer} conformers"
        )
        embeddings = embeddings.reshape(len(smiles_list), n_confomer, -1).mean(axis=1)

    return embeddings.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="path to refined_ESOL.csv")
    p.add_argument("--out", type=Path, default=Path("data/cache"), help="output cache dir")
    p.add_argument("--split-frac", type=float, default=0.1, help="val fraction if no VALID col")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--remove-hs", action="store_true", help="remove hydrogens before encoding")
    p.add_argument("--smiles-col", type=str, default=None,
                   help="explicit SMILES column name (auto-detected if omitted)")
    p.add_argument("--target-col", type=str, default=None,
                   help="explicit target column name (auto-detected if omitted)")
    p.add_argument("--n-confomer", type=int, default=1, help="conformers per molecule (k)")
    p.add_argument("--batch-size", type=int, default=32, help="batch size for Unimol forward")
    p.add_argument("--data-type", type=str, default="molecule",
                   choices=["molecule", "protein", "crystal", "oled"])
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[00] loading ESOL from {args.csv}")
    train_df, val_df = load_split(
        args.csv, args.split_frac, args.seed,
        smiles_col=args.smiles_col, target_col=args.target_col,
    )

    print(f"[00] extracting train embeddings (N={len(train_df)}) ...")
    X_train = extract_embeddings(
        train_df["SMILES"].tolist(),
        data_type=args.data_type, remove_hs=args.remove_hs,
        batch_size=args.batch_size, n_confomer=args.n_confomer, seed=args.seed,
    )
    y_train = train_df["TARGET"].to_numpy(dtype=np.float32)
    print(f"     X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    print(f"[00] extracting val embeddings (N={len(val_df)}) ...")
    X_val = extract_embeddings(
        val_df["SMILES"].tolist(),
        data_type=args.data_type, remove_hs=args.remove_hs,
        batch_size=args.batch_size, n_confomer=args.n_confomer, seed=args.seed,
    )
    y_val = val_df["TARGET"].to_numpy(dtype=np.float32)
    print(f"     X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Save
    torch.save({"X": torch.from_numpy(X_train), "y": torch.from_numpy(y_train)}, args.out / "esol_train.pt")
    torch.save({"X": torch.from_numpy(X_val), "y": torch.from_numpy(y_val)}, args.out / "esol_val.pt")

    # Metadata + checksum (so downstream scripts can verify they're using same embeddings)
    h = hashlib.sha256()
    h.update(X_train.tobytes())
    h.update(X_val.tobytes())
    meta = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "embed_dim": int(X_train.shape[1]),
        "n_confomer": int(args.n_confomer),
        "remove_hs": bool(args.remove_hs),
        "data_type": str(args.data_type),
        "seed": int(args.seed),
        "split_frac": float(args.split_frac),
        "has_VALID_col": "VALID" in pd.read_csv(args.csv).columns,
        "embedding_sha256": h.hexdigest(),
    }
    (args.out / "esol_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[00] saved to {args.out}/")
    print(f"     meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
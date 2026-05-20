"""
00 - Generate conformer dicts for Step 1's pre-split CSVs.

DOES NOT do its own split — loads the EXACT same train/valid/test split that
Step 1 uses (from data/processed/{dataset}/seed_{N}/{dataset}_{split}.csv).

This ensures Step 1 vs EGGROLL comparison is apples-to-apples on the SAME data.

Output:
    data/cache_B/{dataset}/seed_{N}/
        ├── train_conformers.pt
        ├── valid_conformers.pt
        ├── test_conformers.pt
        └── meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _detect_columns(df: pd.DataFrame, smiles_col: str | None, target_col: str | None):
    """Auto-detect smiles/target columns. Step 1's preprocessed CSVs use 'smiles', 'target'."""
    if smiles_col is None:
        for c in ["smiles", "SMILES", "smi", "Smiles", "SMI"]:
            if c in df.columns:
                smiles_col = c; break
    if target_col is None:
        for c in ["target", "TARGET", "y", "label", "measured",
                  "measured log solubility in mols per litre"]:
            if c in df.columns:
                target_col = c; break
    if smiles_col is None or target_col is None:
        raise ValueError(f"Could not detect smiles/target columns. Got: {list(df.columns)}")
    return smiles_col, target_col


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="esol",
                   help="dataset name (matches Step 1's data/processed/{dataset}/ layout)")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                   help="root of Step 1's pre-split CSVs")
    p.add_argument("--split-seed", type=int, required=True,
                   help="split seed (0..4) — must match what Step 1 was run with")
    p.add_argument("--cache-out", type=Path, default=Path("data/cache_B"),
                   help="output cache root")
    p.add_argument("--remove-hs", action="store_true")
    p.add_argument("--n-confomer", type=int, default=1)
    p.add_argument("--max-atoms", type=int, default=256)
    p.add_argument("--smiles-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)
    args = p.parse_args()

    # Paths matching Step 1
    seed_dir_in = args.processed_dir / args.dataset / f"seed_{args.split_seed}"
    csv_paths = {s: seed_dir_in / f"{args.dataset}_{s}.csv" for s in ("train", "valid", "test")}
    missing = [str(p_) for p_ in csv_paths.values() if not p_.exists()]
    if missing:
        raise FileNotFoundError(
            f"Step 1 pre-split CSVs not found:\n  " + "\n  ".join(missing) +
            f"\nRun Step 1 preprocessing first:\n"
            f"  python scripts/preprocess_data.py --dataset {args.dataset} "
            f"--split-seed {args.split_seed}"
        )

    out_dir = args.cache_out / args.dataset / f"seed_{args.split_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[00] dataset={args.dataset}, split_seed={args.split_seed}")
    print(f"     loading Step 1's pre-split CSVs from {seed_dir_in}/")
    dfs = {}
    for s in ("train", "valid", "test"):
        dfs[s] = pd.read_csv(csv_paths[s])
        print(f"       {s}: {len(dfs[s])} rows, cols={list(dfs[s].columns)}")

    # Detect columns once (same for all splits)
    smiles_col, target_col = _detect_columns(dfs["train"], args.smiles_col, args.target_col)
    print(f"     smiles_col='{smiles_col}', target_col='{target_col}'")

    # Init UniMol dictionary (need for atom token mapping in conformers)
    from unimol_tools.models import UniMolModel
    from unimol_tools.data.conformer import ConformerGen
    print(f"[00] initializing UniMol dictionary")
    model = UniMolModel(output_dim=2, data_type="molecule", remove_hs=args.remove_hs)
    dictionary = model.dictionary
    del model

    conf_gen = ConformerGen(
        method="rdkit_random", mode="fast", n_confomer=args.n_confomer,
        remove_hs=args.remove_hs, dictionary=dictionary,
        max_atoms=args.max_atoms, multi_process=False, seed=args.split_seed,
    )

    for split_name in ("train", "valid", "test"):
        df = dfs[split_name]
        smiles_list = df[smiles_col].tolist()
        targets = df[target_col].to_numpy(dtype=np.float32)

        print(f"[00] generating conformers for {split_name} (N={len(smiles_list)}, k={args.n_confomer}) ...")
        confs, _ = conf_gen.transform(smiles_list)
        if len(confs) != len(smiles_list) * args.n_confomer:
            print(f"     ⚠️  got {len(confs)} conformers (expected {len(smiles_list) * args.n_confomer})")

        conformers_t = [{
            "src_tokens": torch.from_numpy(d["src_tokens"]).long(),
            "src_distance": torch.from_numpy(d["src_distance"]).float(),
            "src_coord": torch.from_numpy(d["src_coord"]).float(),
            "src_edge_type": torch.from_numpy(d["src_edge_type"]).long(),
        } for d in confs]

        torch.save({
            "conformers": conformers_t,
            "targets": torch.from_numpy(targets),
            "smiles": smiles_list,  # for traceability
        }, out_dir / f"{split_name}_conformers.pt")

    meta = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "n_train": len(dfs["train"]),
        "n_valid": len(dfs["valid"]),
        "n_test": len(dfs["test"]),
        "n_confomer": args.n_confomer,
        "remove_hs": args.remove_hs,
        "max_atoms": args.max_atoms,
        "smiles_col_used": smiles_col,
        "target_col_used": target_col,
        "source_step1_csvs": [str(p_) for p_ in csv_paths.values()],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[00] saved to {out_dir}/")
    print(f"     meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
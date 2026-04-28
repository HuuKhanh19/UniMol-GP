#!/usr/bin/env python
"""
Dump per-conformer molecular properties for all molecules in a dataset.

Mục đích: Thu thập data để PHÂN TÍCH PHÂN BỐ trước khi quyết định
chia cây (Forest) trong Plan A — ví dụ:
    - Mol-level invariants (cùng giá trị qua mọi conf của 1 mol):
        n_heavy_atoms, n_total_atoms, mol_weight, logP, qed, sa_score,
        n_rotatable_bonds, tpsa, n_aromatic_rings
    - Conf-level varying (khác nhau giữa K conf của cùng 1 mol):
        energy_raw, energy_dE, energy_rank,
        radius_of_gyration, rmsd_to_lowest, npr1, npr2

Usage:
    python scripts/dump_conformer_properties.py --dataset esol
    python scripts/dump_conformer_properties.py --dataset esol --K 10 --seed 42
    python scripts/dump_conformer_properties.py --dataset esol --include-hs

Output:
    data/properties/{dataset}/seed_{split_seed}/properties_K{K}.csv

Note:
    - Conf generation dùng SAME params như pipeline thật:
      ETKDGv3, MMFF94 minimize (fallback UFF), seed=42.
    - num_atoms được tính cả heavy-only (remove_hs=True default của pipeline)
      và total (incl. H sau AddHs) để bạn so sánh.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdMolDescriptors,
)

# Silence RDKit warnings spam
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# Hook into project structure (script runs from project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import DATASET_NAMES  # noqa: E402
from src.data.datasets import PROCESSED_DIR  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# SA score (RDKit Contrib) — try multiple paths
# ──────────────────────────────────────────────────────────────────────

def _try_import_sascorer():
    """Try to import RDKit's Contrib SA_Score module from common locations."""
    # 1. Direct import
    try:
        from rdkit.Contrib.SA_Score import sascorer  # noqa
        return sascorer
    except ImportError:
        pass

    # 2. From conda env share dir
    candidates = [
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "share", "RDKit", "Contrib", "SA_Score"),
        "/usr/share/RDKit/Contrib/SA_Score",
        "/opt/conda/share/RDKit/Contrib/SA_Score",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "sascorer.py")):
            sys.path.insert(0, path)
            try:
                import sascorer  # noqa
                return sascorer
            except ImportError:
                continue
    return None


_sascorer = _try_import_sascorer()
HAS_SA = _sascorer is not None


# ──────────────────────────────────────────────────────────────────────
# Mol-level properties (1 value per molecule)
# ──────────────────────────────────────────────────────────────────────

def compute_mol_level_props(mol_no_h):
    """
    Compute properties that don't depend on 3D conformation.
    Input: RDKit Mol (without explicit H — same as pipeline standard).
    """
    return {
        "n_heavy_atoms":      mol_no_h.GetNumHeavyAtoms(),
        "mol_weight":         Descriptors.MolWt(mol_no_h),
        "logP":               Crippen.MolLogP(mol_no_h),
        "qed":                QED.qed(mol_no_h),
        "sa_score":           _sascorer.calculateScore(mol_no_h) if HAS_SA else np.nan,
        "n_rotatable_bonds":  Lipinski.NumRotatableBonds(mol_no_h),
        "tpsa":               Descriptors.TPSA(mol_no_h),
        "n_aromatic_rings":   rdMolDescriptors.CalcNumAromaticRings(mol_no_h),
        "n_h_donors":         Lipinski.NumHDonors(mol_no_h),
        "n_h_acceptors":      Lipinski.NumHAcceptors(mol_no_h),
    }


# ──────────────────────────────────────────────────────────────────────
# Conf-level properties (K values per molecule)
# ──────────────────────────────────────────────────────────────────────

def _minimize_energy(mol_h, conf_id):
    """Try MMFF94, fallback UFF. Mirrors inner_smi2coords behavior."""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            mp = AllChem.MMFFGetMoleculeProperties(mol_h)
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=conf_id)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
        if ff is None:
            return np.nan
        ff.Minimize()
        return float(ff.CalcEnergy())
    except Exception:
        return np.nan


def _radius_of_gyration(coords, masses=None):
    """Compute Rg of a 3D structure. coords: (N, 3) array."""
    if coords.shape[0] < 2:
        return np.nan
    if masses is None:
        masses = np.ones(coords.shape[0])
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    diff = coords - com
    rg2 = (masses * (diff ** 2).sum(axis=1)).sum() / masses.sum()
    return float(np.sqrt(rg2))


def _rmsd(c1, c2):
    """Heavy RMSD (no alignment) between 2 conformers' coord arrays."""
    if c1.shape != c2.shape:
        return np.nan
    return float(np.sqrt(((c1 - c2) ** 2).sum(axis=1).mean()))


def _npr_ratios(coords, masses):
    """
    Normalized PMI ratios (NPR1, NPR2) — shape descriptor.
    NPR1 = I1/I3, NPR2 = I2/I3 where I1<=I2<=I3 are principal moments.
    Sphere: (1, 1); Rod: (~0, 1); Disk: (~0.5, 0.5).
    """
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    p = coords - com
    # inertia tensor
    I = np.zeros((3, 3))
    for i in range(coords.shape[0]):
        m = masses[i]
        x, y, z = p[i]
        I[0, 0] += m * (y * y + z * z)
        I[1, 1] += m * (x * x + z * z)
        I[2, 2] += m * (x * x + y * y)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z
    I[1, 0] = I[0, 1]; I[2, 0] = I[0, 2]; I[2, 1] = I[1, 2]
    eig = np.sort(np.linalg.eigvalsh(I))  # ascending: I1<=I2<=I3
    if eig[2] < 1e-10:
        return np.nan, np.nan
    return float(eig[0] / eig[2]), float(eig[1] / eig[2])


def gen_confs_with_props(smi, K, seed):
    """
    Sinh K conformers + minimize + tính conf-level props.
    Mirrors inner_smi2coords logic (ETKDGv3 + MMFF/UFF) NHƯNG KHÔNG sort theo energy.

    Returns:
        list of dicts (1 per conf), atoms array, n_total_atoms (incl. H)
        or (None, None, None) on failure.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    mol_h = AllChem.AddHs(mol)

    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.numThreads = 0
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=K, params=ps))

    # Fallback: random coords
    if len(conf_ids) == 0:
        ps.useRandomCoords = True
        ps.maxAttempts = 1000
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=K, params=ps))

    if len(conf_ids) == 0:
        return None, None, None

    n_total = mol_h.GetNumAtoms()
    masses = np.array([a.GetMass() for a in mol_h.GetAtoms()])
    atoms = np.array([a.GetSymbol() for a in mol_h.GetAtoms()])

    # Minimize + collect coords + energies
    confs_data = []
    for cid in conf_ids:
        e = _minimize_energy(mol_h, cid)
        try:
            coords = mol_h.GetConformer(int(cid)).GetPositions().astype(np.float32)
        except Exception:
            continue
        confs_data.append({"conf_id": int(cid), "energy_raw": e, "coords": coords})

    if len(confs_data) == 0:
        return None, None, None

    # Determine lowest-energy conf for RMSD reference + ΔE
    energies = np.array([d["energy_raw"] for d in confs_data], dtype=float)
    finite = np.isfinite(energies)
    if finite.any():
        e_min = float(np.nanmin(energies))
        ref_idx = int(np.nanargmin(energies))
    else:
        e_min = np.nan
        ref_idx = 0
    ref_coords = confs_data[ref_idx]["coords"]

    # Rank ascending (NaN energies → rank = -1)
    ranks = np.full(len(confs_data), -1, dtype=int)
    valid_idx = np.where(finite)[0]
    if len(valid_idx) > 0:
        sorted_valid = valid_idx[np.argsort(energies[valid_idx])]
        for r, i in enumerate(sorted_valid):
            ranks[i] = r

    # Compute conf-level features
    out = []
    for k, d in enumerate(confs_data):
        c = d["coords"]
        e_raw = d["energy_raw"]
        e_dE = (e_raw - e_min) if (np.isfinite(e_raw) and np.isfinite(e_min)) else np.nan
        rg = _radius_of_gyration(c, masses)
        rmsd_lo = _rmsd(c, ref_coords)
        try:
            npr1, npr2 = _npr_ratios(c, masses)
        except Exception:
            npr1, npr2 = np.nan, np.nan
        out.append({
            "conf_idx":            k,                     # generation order
            "rdkit_conf_id":       d["conf_id"],
            "energy_raw":          float(e_raw) if np.isfinite(e_raw) else np.nan,
            "energy_dE":           float(e_dE) if np.isfinite(e_dE) else np.nan,
            "energy_rank":         int(ranks[k]),
            "radius_of_gyration":  rg,
            "rmsd_to_lowest":      rmsd_lo,
            "npr1":                npr1,
            "npr2":                npr2,
        })
    return out, atoms, n_total


# ──────────────────────────────────────────────────────────────────────
# Data loading helper
# ──────────────────────────────────────────────────────────────────────

def load_split(dataset_name, split_seed):
    seed_dir = os.path.join(PROCESSED_DIR, dataset_name, f"seed_{split_seed}")
    paths = {
        s: os.path.join(seed_dir, f"{dataset_name}_{s}.csv")
        for s in ("train", "valid", "test")
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Data not found at {seed_dir}/. "
            f"Run: python scripts/preprocess_data.py "
            f"--dataset {dataset_name} --split-seed {split_seed}"
        )
    return tuple(pd.read_csv(paths[s]) for s in ("train", "valid", "test"))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_NAMES))
    parser.add_argument("--K", type=int, default=10, help="Số conformers / molecule")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="RDKit ETKDGv3 random seed")
    parser.add_argument("--output-dir", default="data/properties")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: chỉ chạy N molecules đầu của mỗi split")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Dump conformer properties — dataset={args.dataset}, K={args.K}")
    print(f"SA scorer available: {HAS_SA}")
    if not HAS_SA:
        print(f"  (install: rdkit Contrib path setup, hoặc pip install rdkit-pypi)")
    print(f"=" * 60)

    train_df, valid_df, test_df = load_split(args.dataset, args.split_seed)
    print(f"Sizes: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

    if args.limit:
        train_df = train_df.head(args.limit)
        valid_df = valid_df.head(args.limit)
        test_df = test_df.head(args.limit)
        print(f"  ⚠ Debug --limit {args.limit} per split")

    rows = []
    mol_global_idx = 0
    n_failed = 0

    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
            smi = row["smiles"]
            target = row.get("target", np.nan)

            try:
                mol_no_h = Chem.MolFromSmiles(smi)
                if mol_no_h is None:
                    raise ValueError("Cannot parse SMILES")
                mol_props = compute_mol_level_props(mol_no_h)
            except Exception as e:
                print(f"  ⚠ Mol-level failed [{smi[:40]}]: {e}")
                mol_global_idx += 1
                n_failed += 1
                continue

            try:
                conf_props_list, atoms, n_total = gen_confs_with_props(
                    smi, args.K, args.seed
                )
            except Exception as e:
                conf_props_list = None

            if conf_props_list is None or len(conf_props_list) == 0:
                print(f"  ⚠ Conf gen failed [{smi[:40]}]")
                mol_global_idx += 1
                n_failed += 1
                continue

            for cprop in conf_props_list:
                rows.append({
                    "mol_idx":       mol_global_idx,
                    "split":         split_name,
                    "smiles":        smi,
                    "target":        target,
                    "n_total_atoms": n_total,
                    **mol_props,        # mol-level
                    **cprop,            # conf-level
                })
            mol_global_idx += 1

    out_df = pd.DataFrame(rows)

    out_dir = os.path.join(args.output_dir, args.dataset, f"seed_{args.split_seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"properties_K{args.K}.csv")
    out_df.to_csv(out_path, index=False)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Saved → {out_path}")
    print(f"Total rows: {len(out_df)} ({len(out_df) // max(1, args.K)} mols × ~{args.K} confs)")
    print(f"Failed molecules: {n_failed}")
    print(f"{'='*60}")

    if len(out_df) == 0:
        print("No rows — aborting summary.")
        return

    print("\n=== MOL-LEVEL stats (1 row per molecule) ===")
    mol_cols = ["n_heavy_atoms", "n_total_atoms", "mol_weight", "logP", "qed",
                "sa_score", "n_rotatable_bonds", "tpsa", "n_aromatic_rings",
                "n_h_donors", "n_h_acceptors"]
    mol_only = out_df.drop_duplicates("mol_idx")[mol_cols]
    print(mol_only.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())

    print("\n=== CONF-LEVEL stats (all confs, all mols) ===")
    conf_cols = ["energy_raw", "energy_dE", "energy_rank",
                 "radius_of_gyration", "rmsd_to_lowest", "npr1", "npr2"]
    print(out_df[conf_cols].describe(percentiles=[.05, .25, .5, .75, .95])
                            .round(3).to_string())

    print("\n=== CONF-LEVEL spread WITHIN each molecule (std across K confs) ===")
    intra = out_df.groupby("mol_idx")[conf_cols].std().describe(
        percentiles=[.05, .25, .5, .75, .95]).round(3)
    print(intra.to_string())
    print("  → cột nào có std intra ≈ 0 nghĩa là không varied giữa K conf của cùng mol")
    print("  → cột std lớn = thực sự conf-level → ứng cử viên cho 'K → n cây'")


if __name__ == "__main__":
    main()
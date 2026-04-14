"""
Data loading and preprocessing.

Loads raw CSV, cleans SMILES, applies scaffold split.
No dependency on config files — all parameters passed directly.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .splitters import random_scaffold_split
from .datasets import get_dataset_info


def load_config(path: str) -> dict:
    """Load a YAML file. Returns empty dict if file not found."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_raw_data(raw_dir: str, filename: str) -> pd.DataFrame:
    """Load raw CSV file."""
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False
    finally:
        RDLogger.EnableLog('rdApp.*')


def preprocess_dataset(
    df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    task_type: str,
) -> pd.DataFrame:
    """Clean SMILES, remove duplicates, standardize column names."""
    df = df.copy()

    for col, label in [(smiles_column, 'SMILES'), (target_column, 'Target')]:
        if col not in df.columns:
            raise KeyError(f"{label} column '{col}' not found. Available: {list(df.columns)}")

    n0 = len(df)
    df = df.dropna(subset=[smiles_column, target_column])
    if len(df) < n0:
        print(f"  Removed {n0 - len(df)} rows with missing values")

    n0 = len(df)
    df = df[df[smiles_column].apply(validate_smiles)]
    if len(df) < n0:
        print(f"  Removed {n0 - len(df)} invalid SMILES")

    n0 = len(df)
    df = df.drop_duplicates(subset=[smiles_column], keep='first')
    if len(df) < n0:
        print(f"  Removed {n0 - len(df)} duplicate SMILES")

    dtype = int if task_type == 'classification' else float
    df = df[[smiles_column, target_column]].copy()
    df.columns = ['smiles', 'target']
    df['target'] = df['target'].astype(dtype)
    return df


def prepare_dataset(
    dataset_name: str,
    raw_dir: str = "data/raw",
    split_ratio: tuple = (0.8, 0.1, 0.1),
    split_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load, preprocess, and scaffold-split a dataset.

    Args:
        dataset_name: Key in DATASET_REGISTRY.
        raw_dir: Directory with raw CSV files.
        split_ratio: (train, valid, test) fractions.
        split_seed: Random seed for scaffold shuffling.

    Returns:
        (train_df, valid_df, test_df, dataset_info)
    """
    info = get_dataset_info(dataset_name)

    df = load_raw_data(raw_dir, info['file'])
    print(f"Loaded {len(df)} molecules from {info['file']}")

    df = preprocess_dataset(
        df, info['smiles_column'], info['target_column'], info['task_type']
    )
    print(f"After preprocessing: {len(df)} molecules")

    smiles_list = df['smiles'].tolist()
    train_df, valid_df, test_df = random_scaffold_split(
        dataset=df,
        smiles_list=smiles_list,
        random_seed=split_seed,
        ratio_test=split_ratio[2],
        ration_valid=split_ratio[1],
        dataframe=True,
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Split — Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    return train_df, valid_df, test_df, info
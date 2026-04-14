"""
Data splitting utilities for CONAN-SchNet.

Provides random scaffold split and random split for molecular datasets.
Adapted from MolHFCNet repository.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold


# =============================================================================
# Scaffold utilities
# =============================================================================

def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Obtain Bemis-Murcko scaffold from a SMILES string."""
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )


# =============================================================================
# Split functions
# =============================================================================

def random_scaffold_split(
    dataset,
    smiles_list,
    random_seed: int = 8,
    ratio_test: float = 0.1,
    ration_valid: float = 0.1,
    dataframe: bool = False,
):
    """Split dataset by random scaffold grouping.

    Groups molecules by Murcko scaffold, then randomly assigns scaffold
    groups to train/valid/test splits.

    Args:
        dataset: The dataset (DataFrame or indexable object).
        smiles_list: Array of SMILES strings.
        random_seed: Random seed for scaffold shuffling.
        ratio_test: Fraction for test set.
        ration_valid: Fraction for validation set (of non-test portion).
        dataframe: If True, return DataFrame slices; else return tensor-indexed.

    Returns:
        Tuple of (train, valid, test) datasets.
    """
    print('Random scaffold split ...........')
    rng = np.random.RandomState(random_seed)

    # Group molecules by scaffold
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    # Shuffle scaffold groups
    scaffold_keys = list(scaffolds.keys())
    scaffold_keys = rng.permutation(scaffold_keys)
    scaffold_sets = [scaffolds[key] for key in scaffold_keys]

    n_total_valid = int(ration_valid * len(dataset) * (1 - ratio_test))
    n_total_test = int(ratio_test * len(dataset))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    # Verify no overlap
    assert len(set(train_idx) & set(valid_idx)) == 0
    assert len(set(test_idx) & set(valid_idx)) == 0
    total = len(set(train_idx)) + len(set(test_idx)) + len(set(valid_idx))
    assert total == len(smiles_list), 'Total samples do not match'

    print(f'  Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}')

    if dataframe:
        return dataset.iloc[train_idx], dataset.iloc[valid_idx], dataset.iloc[test_idx]
    else:
        import torch
        return (
            dataset[torch.tensor(train_idx)],
            dataset[torch.tensor(valid_idx)],
            dataset[torch.tensor(test_idx)],
        )


def random_split(
    dataset,
    random_seed: int = 8,
    ratio_test: float = 0.1,
    ration_valid: float = 0.1,
):
    """Simple random train/valid/test split.

    Args:
        dataset: DataFrame to split.
        random_seed: Random seed.
        ratio_test: Test fraction.
        ration_valid: Validation fraction (of non-test portion).

    Returns:
        Tuple of (train_df, valid_df, test_df).
    """
    train_val, test_df = train_test_split(
        dataset, test_size=ratio_test, random_state=random_seed
    )
    train_df, valid_df = train_test_split(
        train_val, test_size=ration_valid, random_state=random_seed
    )
    print(f'  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}')
    return train_df, valid_df, test_df


# =============================================================================
# K-Fold Splitter (from MolHFCNet)
# =============================================================================

class Splitter:
    """K-fold cross-validation splitter supporting random, scaffold, and stratified."""

    def __init__(self, split_method: str = '5fold_random', seed: int = 42):
        self.n_splits, self.method = (
            int(split_method.split('fold')[0]),
            split_method.split('_')[-1],
        )
        self.seed = seed
        self.splitter = self._init_split()

    def _init_split(self):
        if self.method == 'random':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method in ('scaffold', 'group'):
            return GroupKFold(n_splits=self.n_splits)
        elif self.method == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
        else:
            raise ValueError(
                f'Unknown split method: {self.n_splits}fold_{self.method}'
            )

    def split(self, data, target=None, group=None):
        return self.splitter.split(data, target, group)
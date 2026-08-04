"""
Scaffold splitting for UniMol-GP.

Bemis-Murcko scaffold grouping, then random assignment of whole scaffold
groups to train/valid/test. Adapted from the MolHFCNet repository.

With ratio_test=0.1 and ration_valid=0.1 the effective split is 81/9/10,
because the valid budget is taken from the non-test portion:
    n_valid = ration_valid * N * (1 - ratio_test) = 0.09 * N
"""

from collections import defaultdict

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Obtain Bemis-Murcko scaffold from a SMILES string."""
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )


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

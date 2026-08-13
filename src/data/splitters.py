"""
Splitting for UniMol-GP.

Two splitters, both driven by the same size budget so their numbers are
directly comparable:

* ``random_scaffold_split`` -- Bemis-Murcko scaffold grouping, then random
  assignment of whole scaffold groups. Adapted from the MolHFCNet repository.
  Test molecules carry scaffolds never seen in training, the out-of-
  distribution setting.
* ``random_split`` -- molecules assigned individually, ignoring scaffolds, so
  a scaffold may appear on both sides. The easier, in-distribution setting.

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

    return _materialise(dataset, smiles_list, train_idx, valid_idx, test_idx,
                        dataframe)


def random_split(
    dataset,
    smiles_list,
    random_seed: int = 8,
    ratio_test: float = 0.1,
    ration_valid: float = 0.1,
    dataframe: bool = False,
):
    """Split dataset uniformly at random, ignoring scaffolds.

    The size budget matches random_scaffold_split, but molecules are drawn one
    at a time instead of in scaffold groups, so the splits land on exactly the
    requested sizes rather than the nearest group boundary.

    Args:
        dataset: The dataset (DataFrame or indexable object).
        smiles_list: Array of SMILES strings; used only for its length here.
        random_seed: Random seed for the permutation.
        ratio_test: Fraction for test set.
        ration_valid: Fraction for validation set (of non-test portion).
        dataframe: If True, return DataFrame slices; else return tensor-indexed.

    Returns:
        Tuple of (train, valid, test) datasets.
    """
    print('Random split ...........')
    rng = np.random.RandomState(random_seed)

    n_total = len(dataset)
    n_total_valid = int(ration_valid * n_total * (1 - ratio_test))
    n_total_test = int(ratio_test * n_total)

    perm = rng.permutation(n_total).tolist()
    test_idx = perm[:n_total_test]
    valid_idx = perm[n_total_test:n_total_test + n_total_valid]
    train_idx = perm[n_total_test + n_total_valid:]

    return _materialise(dataset, smiles_list, train_idx, valid_idx, test_idx,
                        dataframe)


def _materialise(dataset, smiles_list, train_idx, valid_idx, test_idx,
                 dataframe: bool):
    """Check the three index sets partition the data, then slice it out."""
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

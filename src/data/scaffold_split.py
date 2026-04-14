"""
Scaffold Split for Molecular Datasets

Implements Bemis-Murcko scaffold splitting for train/valid/test sets.
This is the standard splitting method for molecular property prediction benchmarks.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Optional
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Get Bemis-Murcko scaffold from SMILES string.
    
    Args:
        smiles: SMILES string of the molecule
        include_chirality: Whether to include chirality in scaffold
        
    Returns:
        Scaffold SMILES string, or empty string if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    except Exception:
        return ""


def generate_scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    random_seed: int = 42,
    include_chirality: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    """
    Generate scaffold-based train/valid/test split.
    
    Molecules are grouped by their Bemis-Murcko scaffolds.
    Scaffolds are then assigned to train/valid/test sets to achieve
    the desired split ratios.
    
    Args:
        smiles_list: List of SMILES strings
        frac_train: Fraction for training set (default: 0.8)
        frac_valid: Fraction for validation set (default: 0.1)
        frac_test: Fraction for test set (default: 0.1)
        random_seed: Random seed for reproducibility
        include_chirality: Include chirality in scaffold
        
    Returns:
        Tuple of (train_indices, valid_indices, test_indices)
    """
    np.random.seed(random_seed)
    
    # Validate fractions
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"
    
    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles, include_chirality)
        scaffold_to_indices[scaffold].append(idx)
    
    # Get list of scaffolds and their sizes
    scaffolds = list(scaffold_to_indices.keys())
    scaffold_sizes = [len(scaffold_to_indices[s]) for s in scaffolds]
    
    # Shuffle scaffolds
    indices = list(range(len(scaffolds)))
    np.random.shuffle(indices)
    scaffolds = [scaffolds[i] for i in indices]
    scaffold_sizes = [scaffold_sizes[i] for i in indices]
    
    # Calculate target sizes
    n_total = len(smiles_list)
    n_train = int(n_total * frac_train)
    n_valid = int(n_total * frac_valid)
    n_test = n_total - n_train - n_valid  # Ensure all samples are assigned
    
    # Assign scaffolds to splits using a balanced approach
    train_indices = []
    valid_indices = []
    test_indices = []
    
    # Sort scaffolds by size (largest first) for better distribution
    sorted_scaffold_indices = np.argsort(scaffold_sizes)[::-1]
    
    for idx in sorted_scaffold_indices:
        scaffold = scaffolds[idx]
        scaffold_indices = scaffold_to_indices[scaffold]
        
        # Calculate current deficit for each split
        train_deficit = n_train - len(train_indices)
        valid_deficit = n_valid - len(valid_indices)
        test_deficit = n_test - len(test_indices)
        
        # Assign to the split with largest deficit that can accommodate
        deficits = [
            (train_deficit, 'train'),
            (valid_deficit, 'valid'),
            (test_deficit, 'test')
        ]
        
        # Sort by deficit (largest first)
        deficits.sort(key=lambda x: -x[0])
        
        # Assign to first split with positive deficit
        assigned = False
        for deficit, split_name in deficits:
            if deficit > 0:
                if split_name == 'train':
                    train_indices.extend(scaffold_indices)
                elif split_name == 'valid':
                    valid_indices.extend(scaffold_indices)
                else:
                    test_indices.extend(scaffold_indices)
                assigned = True
                break
        
        # If all deficits are non-positive, assign to train (overflow)
        if not assigned:
            train_indices.extend(scaffold_indices)
    
    # Ensure minimum samples in valid and test
    # If valid or test is empty, move some from train
    if len(valid_indices) == 0 and len(train_indices) > 0:
        # Move last scaffold from train to valid
        move_count = max(1, int(len(train_indices) * frac_valid))
        valid_indices = train_indices[-move_count:]
        train_indices = train_indices[:-move_count]
        print(f"  Warning: Moved {move_count} samples to valid set")
    
    if len(test_indices) == 0 and len(train_indices) > 0:
        # Move last scaffold from train to test
        move_count = max(1, int(len(train_indices) * frac_test))
        test_indices = train_indices[-move_count:]
        train_indices = train_indices[:-move_count]
        print(f"  Warning: Moved {move_count} samples to test set")
    
    return train_indices, valid_indices, test_indices


def scaffold_split_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame using scaffold splitting.
    
    Args:
        df: Input DataFrame
        smiles_column: Name of the SMILES column
        frac_train: Training fraction
        frac_valid: Validation fraction
        frac_test: Test fraction
        random_seed: Random seed
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    smiles_list = df[smiles_column].tolist()
    
    train_idx, valid_idx, test_idx = generate_scaffold_split(
        smiles_list,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        random_seed=random_seed
    )
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    return train_df, valid_df, test_df


if __name__ == "__main__":
    # Test with example
    test_smiles = [
        "CCO",
        "CCCO", 
        "CCCCO",
        "c1ccccc1",
        "c1ccccc1C",
        "c1ccccc1CC",
        "c1ccc2ccccc2c1",
        "CC(=O)O",
        "CC(=O)OC",
    ]
    
    train_idx, valid_idx, test_idx = generate_scaffold_split(
        test_smiles, 
        frac_train=0.7,
        frac_valid=0.15,
        frac_test=0.15
    )
    
    print(f"Total: {len(test_smiles)}")
    print(f"Train: {len(train_idx)} - {train_idx}")
    print(f"Valid: {len(valid_idx)} - {valid_idx}")
    print(f"Test: {len(test_idx)} - {test_idx}")

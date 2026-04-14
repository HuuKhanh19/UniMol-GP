"""
Data Loading and Preprocessing Module

Handles loading CSV files, preprocessing, and preparing data for UniMol.
"""

import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .scaffold_split import scaffold_split_dataframe


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict, dataset_config: Dict) -> Dict:
    """Merge base config with dataset-specific config."""
    merged = base_config.copy()
    merged.update(dataset_config)
    return merged


def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect SMILES column in DataFrame.
    
    Looks for common SMILES column names or columns containing SMILES-like strings.
    """
    # Common SMILES column names (case-insensitive)
    common_names = ['smiles', 'smi', 'smile', 'canonical_smiles', 'mol', 'molecule']
    
    # Check for exact matches (case-insensitive)
    for col in df.columns:
        if col.lower() in common_names:
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        for name in common_names:
            if name in col_lower:
                return col
    
    # Try to detect by content (look for SMILES-like strings)
    for col in df.columns:
        if df[col].dtype == object:  # String column
            sample = df[col].dropna().head(10)
            # SMILES typically contain C, c, (, ), =, #, etc.
            smiles_chars = set('CcNnOoSsPpFClBrI()[]=#@+-.0123456789')
            is_smiles = all(
                len(str(s)) > 3 and 
                set(str(s)).issubset(smiles_chars) and
                ('C' in str(s) or 'c' in str(s))
                for s in sample
            )
            if is_smiles:
                return col
    
    return None


def detect_target_column(df: pd.DataFrame, smiles_col: str, task_type: str) -> Optional[str]:
    """
    Auto-detect target column in DataFrame.
    
    Args:
        df: DataFrame
        smiles_col: Name of SMILES column (to exclude)
        task_type: 'regression' or 'classification'
    """
    # Common target column names
    common_names = ['target', 'label', 'y', 'activity', 'value', 'measured', 'class', 'pchembl_value']
    
    # Get numeric columns (excluding SMILES)
    candidates = []
    for col in df.columns:
        if col == smiles_col:
            continue
        # Check if column is numeric or can be converted
        try:
            pd.to_numeric(df[col], errors='raise')
            candidates.append(col)
        except:
            pass
    
    if not candidates:
        return None
    
    # Check for exact matches first
    for col in candidates:
        if col.lower() in common_names:
            return col
    
    # For classification, prefer columns with few unique values
    if task_type == 'classification':
        for col in candidates:
            unique_vals = df[col].dropna().nunique()
            if unique_vals <= 10:  # Likely classification
                return col
    
    # Return first numeric candidate if no match
    return candidates[0] if candidates else None


def load_raw_data(
    raw_dir: str,
    filename: str
) -> pd.DataFrame:
    """
    Load raw CSV data file.
    
    Args:
        raw_dir: Directory containing raw files
        filename: Name of the CSV file
        
    Returns:
        DataFrame with raw data
    """
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Print column info for debugging
    print(f"  Columns found: {list(df.columns)}")
    
    return df


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit."""
    from rdkit import Chem
    from rdkit import RDLogger
    
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False
    finally:
        RDLogger.EnableLog('rdApp.*')


def preprocess_dataset(
    df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    task_type: str
) -> pd.DataFrame:
    """
    Preprocess dataset: clean SMILES, handle missing values.
    
    Args:
        df: Input DataFrame
        smiles_column: Name of SMILES column
        target_column: Name of target column
        task_type: 'regression' or 'classification'
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy
    df = df.copy()
    
    # Verify columns exist
    if smiles_column not in df.columns:
        raise KeyError(f"SMILES column '{smiles_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")
    
    # Remove rows with missing SMILES or targets
    initial_size = len(df)
    df = df.dropna(subset=[smiles_column, target_column])
    if len(df) < initial_size:
        print(f"  Removed {initial_size - len(df)} rows with missing values")
    
    # Remove invalid SMILES
    initial_size = len(df)
    valid_mask = df[smiles_column].apply(validate_smiles)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        df = df[valid_mask]
        print(f"  Removed {invalid_count} invalid SMILES")
    
    # Remove duplicate SMILES (keep first)
    dup_count = df.duplicated(subset=[smiles_column], keep='first').sum()
    if dup_count > 0:
        df = df.drop_duplicates(subset=[smiles_column], keep='first')
        print(f"  Removed {dup_count} duplicate SMILES")
    
    # For classification, ensure target is integer
    if task_type == 'classification':
        df[target_column] = df[target_column].astype(int)
    else:
        df[target_column] = df[target_column].astype(float)
    
    # Rename columns to standard names for consistency
    df = df[[smiles_column, target_column]].copy()
    df.columns = ['smiles', 'target']
    
    return df


def prepare_dataset(
    dataset_name: str,
    base_config_path: str = "configs/base.yaml",
    dataset_config_dir: str = "configs/datasets",
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load, preprocess, and split a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'esol', 'bace')
        base_config_path: Path to base config
        dataset_config_dir: Directory with dataset configs
        random_seed: Random seed for splitting
        
    Returns:
        Tuple of (train_df, valid_df, test_df, config)
    """
    # Load configs
    base_config = load_config(base_config_path)
    dataset_config_path = os.path.join(dataset_config_dir, f"{dataset_name}.yaml")
    dataset_config = load_config(dataset_config_path)
    
    config = merge_configs(base_config, dataset_config)
    
    # Load raw data
    raw_dir = config['data']['raw_dir']
    filename = config['dataset']['file']
    df = load_raw_data(raw_dir, filename)
    
    print(f"Loaded {len(df)} molecules from {filename}")
    
    # Get column names from config
    smiles_col = config['dataset'].get('smiles_column', None)
    target_col = config['dataset'].get('target_column', None)
    task_type = config['dataset']['task_type']
    
    # Auto-detect columns if not specified or not found
    if smiles_col is None or smiles_col not in df.columns:
        detected = detect_smiles_column(df)
        if detected:
            print(f"  Auto-detected SMILES column: '{detected}'")
            smiles_col = detected
            config['dataset']['smiles_column'] = detected
        else:
            raise ValueError(f"Could not detect SMILES column. Available: {list(df.columns)}")
    
    if target_col is None or target_col not in df.columns:
        detected = detect_target_column(df, smiles_col, task_type)
        if detected:
            print(f"  Auto-detected target column: '{detected}'")
            target_col = detected
            config['dataset']['target_column'] = detected
        else:
            raise ValueError(f"Could not detect target column. Available: {list(df.columns)}")
    
    # Preprocess
    df = preprocess_dataset(df, smiles_col, target_col, task_type)
    print(f"After preprocessing: {len(df)} molecules")
    
    # Scaffold split (using standardized column names now)
    split_ratio = config['data']['split_ratio']
    train_df, valid_df, test_df = scaffold_split_dataframe(
        df,
        smiles_column='smiles',  # Standardized name
        frac_train=split_ratio[0],
        frac_valid=split_ratio[1],
        frac_test=split_ratio[2],
        random_seed=random_seed
    )
    
    print(f"Split sizes - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Update config with standardized column names
    config['dataset']['smiles_column'] = 'smiles'
    config['dataset']['target_column'] = 'target'
    
    return train_df, valid_df, test_df, config


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str
):
    """Save train/valid/test splits to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(output_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.csv")
    
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits to {output_dir}/")
    
    return train_path, valid_path, test_path

"""
Add this class to the END of your existing src/data/data_loader.py file
"""

class DatasetLoader:
    """
    DatasetLoader class for consistent data loading across steps.
    
    Handles loading raw data, preprocessing, and scaffold splitting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DatasetLoader.
        
        Args:
            config: Configuration dictionary with data and dataset settings
        """
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.raw_dir = config['data']['raw_dir']
        self.processed_dir = config['data']['processed_dir']
        self.split_ratio = config['data']['split_ratio']
        self.random_seed = config['data'].get('random_seed', 42)
        
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split dataset.
        
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        # Check if processed splits already exist
        processed_path = os.path.join(self.processed_dir, self.dataset_name)
        train_path = os.path.join(processed_path, 'train.csv')
        valid_path = os.path.join(processed_path, 'valid.csv')
        test_path = os.path.join(processed_path, 'test.csv')
        
        if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
            print(f"Loading preprocessed data from {processed_path}")
            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path)
            test_df = pd.read_csv(test_path)
            return train_df, valid_df, test_df
        
        # Load and preprocess from raw data
        print(f"Loading raw data for {self.dataset_name}...")
        filename = self.config['dataset']['file']
        df = load_raw_data(self.raw_dir, filename)
        
        # Get column names
        smiles_col = self.config['dataset'].get('smiles_column')
        target_col = self.config['dataset'].get('target_column')
        task_type = self.config['dataset']['task_type']
        
        # Auto-detect columns if not specified
        if smiles_col is None or smiles_col not in df.columns:
            smiles_col = detect_smiles_column(df)
            if smiles_col is None:
                raise ValueError(f"Could not detect SMILES column")
            print(f"  Auto-detected SMILES column: '{smiles_col}'")
        
        if target_col is None or target_col not in df.columns:
            target_col = detect_target_column(df, smiles_col, task_type)
            if target_col is None:
                raise ValueError(f"Could not detect target column")
            print(f"  Auto-detected target column: '{target_col}'")
        
        # Preprocess
        df = preprocess_dataset(df, smiles_col, target_col, task_type)
        print(f"After preprocessing: {len(df)} molecules")
        
        # Scaffold split
        train_df, valid_df, test_df = scaffold_split_dataframe(
            df,
            smiles_column='smiles',
            frac_train=self.split_ratio[0],
            frac_valid=self.split_ratio[1],
            frac_test=self.split_ratio[2],
            random_seed=self.random_seed
        )
        
        print(f"Split sizes - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        
        # Save processed splits
        os.makedirs(processed_path, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved processed splits to {processed_path}")
        
        return train_df, valid_df, test_df
    
    def get_smiles_column(self) -> str:
        """Get the SMILES column name (standardized to 'smiles')."""
        return 'smiles'
    
    def get_target_column(self) -> str:
        """Get the target column name (standardized to 'target')."""
        return 'target'

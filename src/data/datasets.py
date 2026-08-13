"""
Dataset registry and fixed project constants.
These are facts / project-level settings, not tunable.
"""

import os

# ── Fixed project constants (shared across all steps) ────────────────────

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "experiments"
SPLIT_RATIO = (0.8, 0.1, 0.1)

#: How train/valid/test are carved out. 'scaffold' is the headline protocol;
#: 'random' is the easier in-distribution control.
SPLIT_TYPES = ('scaffold', 'random')
DEFAULT_SPLIT = 'scaffold'

# ── Dataset metadata ─────────────────────────────────────────────────────

DATASET_REGISTRY = {
    'esol': {
        'file': 'refined_ESOL.csv',
        'smiles_column': 'smiles',
        'target_column': 'measured',
        'task_type': 'regression',
        'metric': 'rmse',
    },
    'freesolv': {
        'file': 'refined_FreeSolv.csv',
        'smiles_column': 'smiles',
        'target_column': 'measured',
        'task_type': 'regression',
        'metric': 'rmse',
    },
    'lipo': {
        'file': 'refined_Lipophilicity.csv',
        'smiles_column': 'smiles',
        'target_column': 'measured',
        'task_type': 'regression',
        'metric': 'rmse',
    },
    'bace': {
        'file': 'refined_BACE.csv',
        # refined_BACE.csv ships ['CID', 'SMILES', 'class'] -- capitalised,
        # unlike the other three files.
        'smiles_column': 'SMILES',
        'target_column': 'class',
        'task_type': 'classification',
        'metric': 'auc',
    },
}

DATASET_NAMES = list(DATASET_REGISTRY.keys())


# The split family is the top path component everywhere -- processed CSVs,
# runs and logs -- so a whole experiment family is one directory that can be
# copied, archived or deleted on its own, and no results table can silently mix
# scaffold numbers with random ones.


def split_dir(split: str, dataset_name: str, split_seed: int) -> str:
    """Directory holding one dataset's train/valid/test CSVs."""
    return os.path.join(PROCESSED_DIR, split, dataset_name,
                        f'seed_{split_seed}')


def experiment_name(split: str, step: str, dataset_name: str,
                    split_seed: int, timestamp: str = '') -> str:
    """Run directory, relative to OUTPUT_DIR."""
    tail = [timestamp] if timestamp else []
    return os.path.join(split, step, dataset_name,
                        f'seed_{split_seed}', *tail)


def get_dataset_info(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {DATASET_NAMES}")
    info = DATASET_REGISTRY[name].copy()
    info['name'] = name
    return info

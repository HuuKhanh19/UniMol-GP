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


def _split_parts(split: str, split_seed: int) -> list[str]:
    """Path components that identify one split of one dataset.

    The scaffold split keeps the original flat ``seed_{n}`` layout so the
    existing processed CSVs, Step 1 checkpoints and published scaffold numbers
    all still resolve at the paths they were written to. Every other split type
    gets its own subtree, so the two families can never be mixed up in a
    results table.
    """
    parts = [] if split == DEFAULT_SPLIT else [split]
    return parts + [f'seed_{split_seed}']


def split_dir(dataset_name: str, split_seed: int,
              split: str = DEFAULT_SPLIT) -> str:
    """Directory holding one dataset's train/valid/test CSVs."""
    return os.path.join(PROCESSED_DIR, dataset_name,
                        *_split_parts(split, split_seed))


def experiment_name(step: str, dataset_name: str, split_seed: int,
                    split: str = DEFAULT_SPLIT, timestamp: str = '') -> str:
    """Run directory, relative to OUTPUT_DIR, matching the split layout."""
    tail = [timestamp] if timestamp else []
    return os.path.join(step, dataset_name,
                        *_split_parts(split, split_seed), *tail)


def get_dataset_info(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {DATASET_NAMES}")
    info = DATASET_REGISTRY[name].copy()
    info['name'] = name
    return info

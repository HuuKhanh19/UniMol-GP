"""
Dataset registry and fixed project constants.
These are facts / project-level settings, not tunable.
"""

# ── Fixed project constants (shared across all steps) ────────────────────

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "experiments"
SPLIT_RATIO = (0.8, 0.1, 0.1)

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
        'smiles_column': 'smiles',
        'target_column': 'class',
        'task_type': 'classification',
        'metric': 'auc',
    },
}

DATASET_NAMES = list(DATASET_REGISTRY.keys())


def get_dataset_info(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {DATASET_NAMES}")
    info = DATASET_REGISTRY[name].copy()
    info['name'] = name
    return info
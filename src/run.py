"""Launcher for EGGROLL fine-tuning of UniMol v1 on ESOL.
Edit the paths + hyperparameters below, then run:
    python run.py
No command-line arguments -- this avoids PowerShell's argument-quoting and
line-continuation quirks entirely (PowerShell does NOT use '\' to continue a
line; that is Bash syntax).
"""
import os
import sys
# ===========================================================================
# 0. IMPORT PATH + hyperscalees bootstrap
# ---------------------------------------------------------------------------
# run.py, unimol_jax.py, esol_pipeline.py, train_eggroll.py and
# _hs_bootstrap.py all sit together in one folder. The `hyperscalees` package
# does NOT have to be a direct sibling: it is auto-located, including the
# common src-layout clone where the real package is at
# <repo>/src/hyperscalees/ rather than just <repo>/.
# ===========================================================================
_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from _hs_bootstrap import bootstrap_hyperscalees

# Leave HYPERSCALEES_DIR blank to auto-detect. Set it ONLY if auto-detect
# fails -- it may point at the package dir OR at the repo that contains it.
#   e.g. r"C:\Users\BKAI\ducluong\DrugOptimization\UniMol-GP\src\hyperscalees"
HYPERSCALEES_DIR = r""
try:
    if HYPERSCALEES_DIR:
        _hs_pkg = bootstrap_hyperscalees(hs_dir=HYPERSCALEES_DIR)
    else:
        _hs_pkg = bootstrap_hyperscalees(search_from=_src)
except ImportError as e:
    raise SystemExit(f"[run.py] {e}")
print(f"[run.py] hyperscalees package: {_hs_pkg}")

from train_eggroll import run_finetuning, TrainConfig
from esol_pipeline import load_esol_csv
# ===========================================================================
# 1. PATHS  -- edit these. Windows paths: keep the r"..." (raw string) prefix.
# ===========================================================================
CHECKPOINT = r"C:\Users\BKAI\ducluong\DrugOptimization\Unimol-GP\unimol_source\unimol_tools\weights\mol_pre_all_h_220816.pt"
DICT_FILE  = r"C:\Users\BKAI\ducluong\DrugOptimization\Unimol-GP\unimol_source\unimol_tools\weights\mol.dict.txt"
TRAIN_CSV  = r"C:\Users\BKAI\ducluong\DrugOptimization\Unimol-GP\data\processed\esol\seed_0\esol_train.csv"
TEST_CSV   = r"C:\Users\BKAI\ducluong\DrugOptimization\Unimol-GP\data\processed\esol\seed_0\esol_test.csv"
# CSV column names
SMILES_COL = "smiles"
TARGET_COL = "target"
# ===========================================================================
# 2. HYPERPARAMETERS
# ===========================================================================
cfg = TrainConfig(
    n_epochs    = 25,    # 2000
    n_pop       = 32,     # 256 population size (must be even)
    pop_chunk   = 4,       # GPU-memory chunk; if you hit OOM lower this (2, then 1)
    sigma       = 0.01,    # ES perturbation scale
    lr          = 1e-3,    # Adam learning rate
    rank        = 1,       # EGGROLL probe rank
    target_rmse = 0.78,    # baseline to reach
    log_every   = 1,
)
# Optional: widen the trainable adapters if test RMSE plateaus above target.
# MODEL_OVERRIDES = {"lora_targets": ("q_proj", "k_proj", "v_proj",
#                                     "out_proj", "fc1", "fc2"), "r_lora": 8}
MODEL_OVERRIDES = None
# ===========================================================================
# 3. RUN
# ===========================================================================
if __name__ == "__main__":
    train_smiles, train_y = load_esol_csv(
        TRAIN_CSV, smiles_col=SMILES_COL, target_col=TARGET_COL)
    test_smiles, test_y = load_esol_csv(
        TEST_CSV, smiles_col=SMILES_COL, target_col=TARGET_COL)
    print(f"train: {len(train_smiles)} molecules | test: {len(test_smiles)} molecules")
    params, history, scaler = run_finetuning(
        CHECKPOINT, DICT_FILE,
        train_smiles, train_y, test_smiles, test_y,
        train_config=cfg, model_overrides=MODEL_OVERRIDES)
    print(f"\nbest test RMSE: {history[-1]['best_test_rmse']:.4f}"
          f"   (target ~{cfg.target_rmse})")
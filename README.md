# UniMol-GP — UniMol v1 baseline

Molecular property prediction with UniMol v1, fine-tuned by gradient descent on a
Bemis-Murcko scaffold split.

## Layout

```
config.yaml              tunable hyperparameters (CLI > config.yaml > script defaults)
data/raw/                refined_*.csv source files (not tracked)
data/processed/          scaffold splits, one dir per dataset/seed (not tracked)
experiments/             training runs + results.json (not tracked)
scripts/
  preprocess_data.py     raw CSV -> cleaned + scaffold split -> data/processed/
  run_step1.py           UniMol v1 fine-tuning + evaluation
src/
  data/datasets.py       dataset registry and project constants
  data/splitters.py      Bemis-Murcko scaffold split
  data/data_loader.py    load / clean / split pipeline
  models/unimol_wrapper.py   MolTrain + MolPredict wrapper, Step1Trainer
  utils/helpers.py       timer, JSON I/O, console output
unimol_source/           PATCHED fork of Uni-Mol tools (see note below)
```

## Setup

```bash
conda activate conan_es
bash setup.sh
```

`setup.sh` installs `unimol_source/` in editable mode plus `requirements.txt`.
PyTorch is not installed automatically — install the build matching your CUDA
version first.

## Usage

```bash
# 1. Scaffold-split the raw data (one dir per seed)
python scripts/preprocess_data.py --dataset all --split-seed 0 1 2 3 4

# 2. Train + evaluate one seed
python scripts/run_step1.py --dataset esol --split-seed 0

# Overrides: CLI beats config.yaml beats DEFAULTS in the script
python scripts/run_step1.py --dataset esol --split-seed 2 --epochs 50 --gpu-id 1
```

Results land in `experiments/step1/{dataset}/seed_{X}/{timestamp}/results.json`
together with the checkpoint, so a 5-seed mean is just an average over the five
`seed_*` runs.

## Split

`src/data/splitters.py:random_scaffold_split` groups molecules by Murcko scaffold
(with chirality) and assigns whole scaffold groups to test, then valid, then
train. With `SPLIT_RATIO = (0.8, 0.1, 0.1)` the effective ratio is **81/9/10**,
because the valid budget is taken from the non-test portion:

```
n_test  = 0.1 * N
n_valid = 0.1 * N * (1 - 0.1) = 0.09 * N
n_train = remainder            = 0.81 * N
```

This is intentional — do not "fix" it to 80/10/10.

## Datasets

| key | file | target | task | metric |
|---|---|---|---|---|
| `esol` | `refined_ESOL.csv` | `measured` | regression | rmse |
| `freesolv` | `refined_FreeSolv.csv` | `measured` | regression | rmse |
| `lipo` | `refined_Lipophilicity.csv` | `measured` | regression | rmse |
| `bace` | `refined_BACE.csv` | `class` | classification | auc |

## Note on `unimol_source/`

This is [Uni-Mol tools](https://github.com/deepmodeling/Uni-Mol) **0.1.4,
byte-identical to upstream except for one additive patch**, so the model,
featurisation and training loop are stock UniMol v1.

The single deviation is `MolTrain._override_split_with_valid_column()` in
`unimol_tools/train.py` (~35 added lines): if the input CSV carries a `VALID`
column (0 = train, 1 = valid), it replaces `split_nfolds` with that one fold so
the scaffold split from `preprocess_data.py` is used verbatim. Without a `VALID`
column it is a no-op. Nothing else — no k-fold removal, no featurisation change.

To verify the fork is still clean:

```bash
pip download unimol_tools==0.1.4 --no-deps --no-binary :all: -d /tmp/um
tar xzf /tmp/um/unimol_tools-0.1.4.tar.gz -C /tmp/um
diff -r /tmp/um/unimol_tools-0.1.4/unimol_tools unimol_source/unimol_tools
```

Only `train.py` should differ (plus `config/default.yaml` and
`weights/mol.dict.txt`, which the sdist does not ship).

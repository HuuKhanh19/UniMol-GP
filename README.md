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

# Every knob is a flag; --help lists them with their effective defaults
python scripts/run_step1.py --dataset esol --split-seed 2 --epochs 50 --gpu-id 1
python scripts/run_step1.py --help
```

### Configuration

`config.yaml` is folded into the argparse defaults with `set_defaults`, so the
precedence is:

```
CLI flag  >  config.yaml  >  the default declared in add_argument()
```

Each option therefore has exactly one declared default, and `--help` prints the
value that would actually be used. Keys in `config.yaml` must match an argparse
destination — an unknown key aborts the run and lists the valid ones, so a typo
or a stale key fails loudly instead of being silently ignored:

```
run_step1.py: error: unknown key(s) in config: lrate, n_confomer
allowed: batch_size, epochs, freeze_layers, gpu_id, learning_rate, ...
```

Boolean flags are single-direction (`--no-gpu`, `--no-amp`, `--remove-hs`) and
write to the positive destination (`use_gpu`, `use_amp`, `remove_hs`), which is
also the name to use in `config.yaml`.

Results land in `experiments/step1/{dataset}/seed_{X}/{timestamp}/results.json`
together with the checkpoint, so a 5-seed mean is just an average over the five
`seed_*` runs.

### Hydrogens and the pretrained checkpoint

`remove_hs` picks which pretrained weights get loaded — see
`unimol_tools/models/unimol.py`, `name = "no_h" if remove_hs else "all_h"`:

| `remove_hs` | hydrogens | checkpoint |
|---|---|---|
| `false` (this project's setting) | kept | `mol_pre_all_h_220816.pt` |
| `true` | stripped | `mol_pre_no_h_220816.pt` |

The dictionary (`mol.dict.txt`) is the same either way; only the checkpoint and
the atom list change. `false` is also the unimol_tools default. Keeping the
hydrogens means more atoms per molecule, so runs are slower and results are not
comparable with no-H runs.

### Which unimol_tools is actually running

`run_step1.py` resolves `unimol_tools.__file__` before training and aborts if it
points outside this checkout, then prints the path in the run header. A second
clone of this repo sharing one conda env is otherwise indistinguishable at
runtime — `git pull` updates the files, but the installed package keeps pointing
at wherever `pip install -e` was last run:

```
unimol_tools resolves to:
    C:\...\DrugOptimization\Final\UniMol-GP\unimol_source\unimol_tools
but this repo ships its own patched fork at:
    C:\...\DrugOptimization\UniMol-GP\unimol_source
```

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

# UniMol-GP — UniMol v1 baseline

Molecular property prediction with UniMol v1, fine-tuned by gradient descent on a
Bemis-Murcko scaffold split.

## Layout

```
data/raw/                refined_*.csv source files (not tracked)
data/processed/          splits, {split}/{dataset}/seed_{n}/ (not tracked)
experiments/             runs + results.json, {split}/step{1,2}/... (not tracked)
scripts/
  preprocess_data.py     raw CSV -> cleaned + split -> data/processed/
  run_step1.py           UniMol v1 fine-tuning + evaluation
  run_step2.py           symbolic GP head + EGGROLL fine-tuning
  selftest.py            correctness checks -- run before any step-2 training
src/
  data/datasets.py       dataset registry and project constants
  data/splitters.py      Bemis-Murcko scaffold split, and plain random split
  data/data_loader.py    load / clean / split pipeline
  models/unimol_wrapper.py   MolTrain + MolPredict wrapper, Step1Trainer
  head/                  symbolic head: postfix trees, GP, ridge merge
    ops.py               protected operator set and opcodes
    genome.py            postfix genome, init, crossover, mutation
    evaluator.py         batched lock-step tree evaluation
    ridge.py             batched ridge + scaffold-grouped CV scoring
    gp.py                cooperative coevolution over k islands
    gp_head.py           GPHead container, regions, linear probe
  es/                    EGGROLL: low-rank evolution strategies
    forward_unimol.py    frozen prefix / perturbed suffix split
    perturb.py           antithetic low-rank sampling, aggregation, Adam
    shaping.py           fitness shaping
    data.py              featurisation, length tiling, scaffold folds
    eggroll.py           the ES step
  train/stage2.py        ES <-> GP alternation
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
# 1. Split the raw data (one dir per seed)
python scripts/preprocess_data.py --dataset all --split-seed 0 1 2 3 4
python scripts/preprocess_data.py --dataset all --split random --split-seed 0 1 2 3 4

# 2. Train + evaluate one seed
python scripts/run_step1.py --dataset esol --split-seed 0

# Every knob is a flag; --help is the full reference
python scripts/run_step1.py --dataset esol --split-seed 2 --epochs 50 --gpu-id 1
python scripts/run_step1.py --help
```

### Configuration

**Argparse only — no config file, no Hydra.** `add_argument` is the single
declaration site for every option and its default, so `--help` is the complete
and authoritative reference; there is no second place a value can come from and
no precedence order to reason about. To change a default permanently, edit the
`add_argument` call.

Options are grouped (`data split`, `training`, `featurisation`, `hardware`,
`output`) so `--help` reads as documentation. Boolean flags are
single-direction — `--no-gpu`, `--no-amp`, `--remove-hs` — and write to the
positive destination (`use_gpu`, `use_amp`, `remove_hs`).

For repeatable runs, put the flags in a shell script or record the exact command
in your notes; `results.json` also stores the fully resolved `params` of every
run, so any result can be traced back to the settings that produced it.

Results land in `experiments/{split}/step1/{dataset}/seed_{X}/{timestamp}/results.json`
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

## Step 2 — symbolic head + EGGROLL

Step 2 replaces UniMol's head with **k formula trees over disjoint regions of the
512-d CLS embedding, merged by ridge**, and fine-tunes the last transformer
blocks with **low-rank evolution strategies** (EGGROLL) instead of gradient
descent.

```bash
# always run this first -- it verifies the rewritten forward pass
python scripts/selftest.py

python scripts/run_step2.py --dataset esol --split-seed 0 \
    --init-checkpoint experiments/scaffold/step1/esol/seed_0/<timestamp>/model_0.pth
```

For a full sweep, `scripts/run_all_seeds.ps1` runs one (dataset, seed) after
another, resolving each one's Step 1 checkpoint itself and writing one log per
run. It verifies the self-test, every processed split and every checkpoint
*before* the first run starts, so a missing file fails in the first minute
rather than after hours of GPU time, and it prints a per-run valid/test summary
at the end:

```powershell
.\scripts\run_all_seeds.ps1                                # 5 seeds, GPU 0
.\scripts\run_all_seeds.ps1 -Seeds '0,2,4' -GpuId 0        # split across both
.\scripts\run_all_seeds.ps1 -Seeds '1,3'   -GpuId 1        #   GPUs, run twice

# the whole random-split matrix; lipo costs about as much as the other
# three together, so it gets a card to itself
.\scripts\run_all_seeds.ps1 -Split random -Dataset 'lipo' -GpuId 0
.\scripts\run_all_seeds.ps1 -Split random -Dataset 'esol,freesolv,bace' -GpuId 1
```

A seed with no Step 1 checkpoint is skipped rather than quietly started from the
pretrained weights -- that is a different, much harder experiment and its numbers
must not be averaged in with the rest.

Note that UniMol v1's stock head is `LinearHead` — `Dropout -> Linear(512, 1)`,
513 parameters ([`unimol.py`](unimol_source/unimol_tools/models/unimol.py)) —
not an MLP. Step 2 replaces a *linear* map, so the headroom on a frozen
embedding is small by construction; the hypothesis being tested is that ES
co-adaptation of the backbone to a symbolic head beats co-adaptation to a linear
one.

### Structure

```
z = CLS(512)  ->  16 regions of 32 dims  ->  T_1..T_16   ->  ridge  ->  y
                                             (GP)          + linear probe
                        backbone: EGGROLL over the last 4 blocks
```

Three parameter groups, three mechanisms, alternated rather than nested:

| | search space | optimiser |
|---|---|---|
| backbone (last L blocks) | ~12.6M continuous | EGGROLL |
| tree structures | discrete | cooperative-coevolutionary GP |
| merge weights | k + 2 | ridge, closed form |

GP is **not** run inside the ES fitness. That would make fitness a stochastic
function of the backbone — two nearby parameter vectors could yield different
trees — which destroys the local continuity ES needs, quite apart from costing N
GP runs per step. Instead the trees are fixed and shared across the whole
population within an ES step (common random numbers), and the GP phase
warm-starts from the previous population so the head drifts rather than jumps.

### Fitness

**Scaffold-grouped 5-fold CV inside the training split**, for both GP and ES.
With ~900 molecules and 512 features on tap, training error would be optimised
into meaninglessness. The valid split is reserved for phase-level early stopping
only; test is untouched until the end.

Ridge is solved exactly for every candidate rather than searched — variable
projection. This removes k+2 nuisance dimensions from both searches and makes
fitness invariant to rescaling of the tree outputs.

### Why no GPU genetic-programming library

Embeddings are cached per phase, so a GP generation is ~600 kernel launches on
`(P, n)` tensors — a few milliseconds, well under 3% of the step budget, which a
CUDA GP library would not meaningfully improve. The fitness here (batched ridge
with grouped CV under cooperative coevolution) also does not map onto the fused
fitness kernels such libraries provide. `src/head/evaluator.py` is a tensorised
postfix interpreter in plain torch: same idea, no build dependency, full control.

### Knobs that matter

| flag | why |
|---|---|
| `--rank` | keep `es_pop * rank > 512`, or the aggregate ES update is rank-deficient. The EGGROLL paper's r=1 results all satisfy this via large N. |
| `--es-chunk`, `--mol-tile` | attention memory is `~3 * chunk * tile * heads * S^2`; these are the VRAM knobs. The run prints the estimate against the real longest molecule. |
| `--sigma` | relative to each matrix's Frobenius norm, so one value works across 512x512 and 512x2048. |
| `--init-checkpoint` | start the ES mean from a fine-tuned step-1 model. Starting from pretrained weights is a much harder problem. |
| `--probe-penalty-scale` | the linear probe is a safety net that guarantees the head can reproduce the linear baseline; raise this if `share_trees` in the logs shows the trees going vestigial. |
| `--region-mask` | confine each perturbation column to one head region, so ES asks "which region should change". |

### Multi-GPU

There is no distributed code. With 15 runs to do (5 splits x 3 seeds), running
two independent runs concurrently with `--gpu-id 0` and `--gpu-id 1` beats
splitting one run's population: no sync, no NCCL (unavailable on Windows), same
throughput on the sweep.

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

### `--split scaffold` vs `--split random`

`--split random` (`random_split`) draws molecules individually instead of in
scaffold groups, on the same size budget. Test scaffolds are then seen in
training, which makes it the easier, in-distribution control — expect better
numbers than the scaffold protocol, and never compare the two families
directly.

The flag runs the whole way through `preprocess_data.py`, `run_step1.py` and
`run_step2.py`, and picks the paths as well as the splitter:

The split name is the top path component everywhere:

```
data/processed/{split}/{dataset}/seed_{n}/{dataset}_{train,valid,test}.csv
experiments/{split}/step{1,2}/{dataset}/seed_{n}/{timestamp}/
logs/{split}/step2/{timestamp}/
```

So a whole experiment family is one directory that can be copied, archived or
deleted on its own, and no results table can silently mix scaffold numbers with
random ones. `split_dir` and `experiment_name` in `src/data/datasets.py` are the
one place this rule lives; `Get-SeedPath` in `run_all_seeds.ps1` mirrors it.

The inner CV that scores GP and ES (`--n-folds`) stays scaffold-grouped under
both, since its job is to stop the search memorising scaffolds.

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

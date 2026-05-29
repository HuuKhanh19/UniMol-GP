"""EGGROLL training loop for fine-tuning UniMol v1 on ESOL (Task 6).

Ties Tasks 2-5 together. Each epoch is the EGGROLL 5-step cycle:

    1. generate an N-member population        (theta_i = theta + sigma * E_i)
    2. evaluate fitness                       evaluate_population -> f_i = -MSE_i
    3. convert / z-score the fitness           noiser.convert_fitnesses
    4. estimate the gradient  g ~ sum f_i E_i  } noiser.do_updates
    5. step the optimiser (optax Adam)         }

Fitness is the full-batch MSE on the TRAIN split (in standardised target
space). The monitored / reported metric is the real-units RMSE on the TEST
split -- the goal of this phase is to drive test RMSE down to the standard
UniMol-v1 baseline (~0.78 on ESOL). Once that is reproduced, the value of
EGGROLL shows when the architecture is changed (non-differentiable or
otherwise hard-to-backprop components).

Only theta = {LoRA P/Q adapters} u {head} is optimised; the pretrained
encoder stays frozen.

Hyperparameters (handover): ES rank r = 1, sigma ~ 0.01, Adam lr ~ 1e-3,
population 512-2048, noise_reuse = 1 (fresh perturbations every epoch).

GPU MEMORY: vmapping a 512+ population of full UniMol v1 at once OOMs a 16 GB
card. Set TrainConfig.pop_chunk to a small divisor of n_pop (e.g. 8-32) -- the
population is then evaluated in chunks (see esol_pipeline.evaluate_population).

NOTE -- the frozen encoder weight W lives in `params` (es_map EXCLUDED), so
optax allocates unused (always-zero) Adam moments for it: ~376 MB of wasted
optimiser state at full scale. Tolerable on a 16 GB GPU; move W into
`frozen_params` if memory-constrained.
"""
import jax
import jax.numpy as jnp
import optax

from unimol_jax import (
    _load_hs_module, UniMolV1, load_torch_checkpoint,
    transform_torch_model, validate_assembly,
)

simple_es_tree_key = _load_hs_module("hyperscalees.models.common").simple_es_tree_key

try:                                              # real HyperscaleES repo
    EggRoll = _load_hs_module("hyperscalees.noiser.eggroll").EggRoll
except Exception:                                 # standalone stub layout
    EggRoll = _load_hs_module("hyperscalees.models.noiser.eggroll").EggRoll

from esol_pipeline import (
    Dictionary, esol_config, load_esol_csv, featurize_dataset,
    evaluate_population, evaluate_clean,
)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
class TrainConfig:
    """EGGROLL fine-tuning hyperparameters."""

    def __init__(self, n_epochs=2000, n_pop=512, sigma=0.01, lr=1e-3, rank=1,
                 noise_reuse=1, group_size=0, pop_chunk=16, log_every=25,
                 target_rmse=0.78):
        assert n_pop % 2 == 0, "n_pop must be even (EGGROLL antithetic pairs)"
        if pop_chunk is not None:
            assert n_pop % pop_chunk == 0, "n_pop must be divisible by pop_chunk"
        self.n_epochs = n_epochs
        self.n_pop = n_pop            # population size (parallel perturbations)
        self.sigma = sigma            # ES perturbation scale
        self.lr = lr                  # Adam learning rate
        self.rank = rank              # ES probe rank r (EGGROLL low-rank)
        self.noise_reuse = noise_reuse
        self.group_size = group_size  # 0 -> z-score over the whole population
        self.pop_chunk = pop_chunk    # population chunk size for GPU memory
        self.log_every = log_every
        self.target_rmse = target_rmse


def make_noiser(params, tc):
    """EGGROLL noiser with an Adam solver, configured from a TrainConfig."""
    return EggRoll.init_noiser(
        params, sigma=tc.sigma, lr=tc.lr, solver=optax.adam,
        group_size=tc.group_size, freeze_nonlora=False,   # MUST be False
        noise_reuse=tc.noise_reuse, rank=tc.rank)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_test_rmse(noiser, frozen_noiser_params, noiser_params,
                       frozen_params, params, es_tree_key, test_batch, scaler):
    """Real-units RMSE of the current (unperturbed) theta on the test split."""
    _, mse = evaluate_clean(noiser, frozen_noiser_params, noiser_params,
                            frozen_params, params, es_tree_key, test_batch)
    rmse_std = float(mse) ** 0.5                  # MSE is in standardised space
    return scaler.std * rmse_std if scaler is not None else rmse_std


# --------------------------------------------------------------------------
# one EGGROLL epoch
# --------------------------------------------------------------------------
def make_train_step(noiser, frozen_noiser_params, frozen_params,
                    es_tree_key, es_map, train_batch, n_pop, pop_chunk):
    """Build a jitted train_step(noiser_params, params, epoch).

    Static context (noiser, frozen params, es_tree_key, es_map, train_batch,
    n_pop, pop_chunk) is closed over -- only (noiser_params, params, epoch)
    are jit arguments, matching the EGGROLL worked-example pattern.
    """
    def train_step(noiser_params, params, epoch):
        # 1-2: generate population + evaluate full-batch TRAIN MSE fitness
        raw_fitness, mse = evaluate_population(
            noiser, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, epoch, n_pop, train_batch, pop_chunk=pop_chunk)

        # 3: z-score the fitness over the population
        fitness = noiser.convert_fitnesses(
            frozen_noiser_params, noiser_params, raw_fitness)

        # 4-5: score-function gradient -> optax Adam step
        iterinfos = (jnp.full(n_pop, epoch, dtype=jnp.int32),
                     jnp.arange(n_pop, dtype=jnp.int32))
        noiser_params, params = noiser.do_updates(
            frozen_noiser_params, noiser_params, params, es_tree_key,
            fitness, iterinfos, es_map)

        metrics = {"mean_fitness": jnp.mean(raw_fitness),
                   "mean_train_mse": jnp.mean(mse),
                   "best_train_mse": jnp.min(mse)}
        return noiser_params, params, metrics

    return jax.jit(train_step)


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def train(noiser, frozen_noiser_params, noiser_params, frozen_params, params,
          es_tree_key, es_map, train_batch, test_batch=None, *, n_epochs, n_pop,
          scaler, pop_chunk=None, target_rmse=0.78, log_every=25, verbose=True):
    """Run the EGGROLL training loop.

    Fitness is the full-batch MSE on `train_batch`; the monitored metric is
    real-units RMSE on `test_batch` (defaults to `train_batch` if not given,
    e.g. for an overfit smoke test).

    Returns (params, noiser_params, history). The best test RMSE seen is
    tracked, since ES jitters slightly once near convergence.
    """
    if test_batch is None:
        test_batch = train_batch
    step = make_train_step(noiser, frozen_noiser_params, frozen_params,
                           es_tree_key, es_map, train_batch, n_pop, pop_chunk)

    def test_rmse():
        return evaluate_test_rmse(noiser, frozen_noiser_params, noiser_params,
                                  frozen_params, params, es_tree_key,
                                  test_batch, scaler)

    rmse0 = test_rmse()
    best = rmse0
    history = [{"epoch": 0, "test_rmse": rmse0}]
    if verbose:
        print(f"epoch {0:6d} | test RMSE {rmse0:.4f}"
              f"   (baseline; target ~{target_rmse})")

    for epoch in range(n_epochs):
        noiser_params, params, metrics = step(noiser_params, params, epoch)
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            r = test_rmse()
            best = min(best, r)
            history.append({"epoch": epoch + 1, "test_rmse": r,
                            "best_test_rmse": best,
                            "train_mse": float(metrics["mean_train_mse"])})
            if verbose:
                hit = "  <-- target reached" if r <= target_rmse else ""
                print(f"epoch {epoch + 1:6d} | test RMSE {r:.4f}  "
                      f"(best {best:.4f}){hit}")

    if verbose:
        print(f"\nbest test RMSE: {best:.4f}   (target ~{target_rmse})")
    return params, noiser_params, history


# --------------------------------------------------------------------------
# end-to-end: pretrained checkpoint + ESOL train/test splits -> fine-tuned theta
# --------------------------------------------------------------------------
def run_finetuning(checkpoint_path, dict_path,
                   train_smiles, train_targets, test_smiles, test_targets,
                   train_config=None, model_overrides=None, seed=0):
    """Full pipeline. The TargetScaler is fitted on the TRAIN targets and
    reused for the TEST split, so test RMSE is reported in real units.

    Returns (fine_tuned_params, history, scaler).
    """
    tc = train_config or TrainConfig()

    # 1. dictionary -- MUST be the official mol.dict.txt matching the checkpoint
    dictionary = Dictionary.load(dict_path)

    # 2. model config wired to the dictionary
    cfg = esol_config(dictionary, **(model_overrides or {}))

    # 3. load + convert the pretrained checkpoint  (Task 3)
    state_dict = load_torch_checkpoint(checkpoint_path)
    weights = transform_torch_model(state_dict, cfg)

    # 4. assemble: frozen encoder + LoRA adapters + fresh head  (Tasks 2/4)
    key = jax.random.key(seed)
    frozen_params, params, scan_map, es_map = UniMolV1.rand_init(
        jax.random.fold_in(key, 0), weights, cfg)
    validate_assembly(params, es_map)

    # 5. featurise ESOL  (Task 5) -- scaler fitted on TRAIN, reused on TEST
    train_batch, scaler = featurize_dataset(
        train_smiles, train_targets, dictionary, dtype=cfg.dtype, standardize=True)
    test_batch, _ = featurize_dataset(
        test_smiles, test_targets, dictionary, dtype=cfg.dtype, scaler=scaler)

    # 6. es_tree_key + EGGROLL noiser
    es_tree_key = simple_es_tree_key(params, jax.random.fold_in(key, 1), scan_map)
    frozen_noiser_params, noiser_params = make_noiser(params, tc)

    # 7. train  (Task 6) -- fitness on train, monitor test RMSE
    params, noiser_params, history = train(
        EggRoll, frozen_noiser_params, noiser_params, frozen_params, params,
        es_tree_key, es_map, train_batch, test_batch,
        n_epochs=tc.n_epochs, n_pop=tc.n_pop, scaler=scaler,
        pop_chunk=tc.pop_chunk, target_rmse=tc.target_rmse,
        log_every=tc.log_every)
    return params, history, scaler


def _read_smiles_targets(csv_path, smiles_col, target_col):
    if smiles_col is None and target_col is None:
        return load_esol_csv(csv_path)
    return load_esol_csv(csv_path, smiles_col=smiles_col, target_col=target_col)


def main():
    """Entry point for a real run. Supply the official UniMol v1 checkpoint,
    the matching mol.dict.txt, and ESOL train/test CSVs."""
    import argparse
    ap = argparse.ArgumentParser(description="EGGROLL fine-tune UniMol v1 on ESOL")
    ap.add_argument("--checkpoint", required=True, help="UniMol v1 .pt checkpoint")
    ap.add_argument("--dict", required=True, help="official mol.dict.txt")
    ap.add_argument("--train-csv", required=True, help="ESOL train split CSV")
    ap.add_argument("--test-csv", required=True, help="ESOL test split CSV")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--target-col",
                    default="measured log solubility in mols per litre")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--pop-chunk", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=1)
    args = ap.parse_args()

    train_smiles, train_y = _read_smiles_targets(
        args.train_csv, args.smiles_col, args.target_col)
    test_smiles, test_y = _read_smiles_targets(
        args.test_csv, args.smiles_col, args.target_col)

    tc = TrainConfig(n_epochs=args.epochs, n_pop=args.pop, sigma=args.sigma,
                     lr=args.lr, rank=args.rank, pop_chunk=args.pop_chunk)
    params, history, scaler = run_finetuning(
        args.checkpoint, args.dict, train_smiles, train_y, test_smiles, test_y,
        train_config=tc)
    print(f"\nfinished -- best test RMSE {history[-1]['best_test_rmse']:.4f}")


if __name__ == "__main__":
    main()
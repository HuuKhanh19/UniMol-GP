"""smoke_test.py -- verify the EGGROLL/UniMol-v1 pipeline runs end-to-end on CPU.

No torch, no pretrained checkpoint, no CSVs required: a tiny UniMol-v1 is
built with random weights (make_random_weights) and a handful of fake
molecules (featurize_from_coords -- no RDKit). It exercises exactly the code
path run.py uses: assembly -> es_map validation -> EGGROLL noiser -> training
loop. Run from the folder holding the deliverables + the hyperscalees package:

    python smoke_test.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from unimol_jax import (UniMolV1, make_random_weights, validate_assembly,
                        theta_summary, _load_hs_module)
from esol_pipeline import (build_test_dictionary, esol_config,
                           featurize_from_coords, build_batch, TargetScaler)
from train_eggroll import TrainConfig, make_noiser, train

simple_es_tree_key = _load_hs_module("hyperscalees.models.common").simple_es_tree_key
EggRoll = _load_hs_module("hyperscalees.noiser.eggroll").EggRoll

key = jax.random.key(0)

# --- 1. tiny model -------------------------------------------------------
dictionary = build_test_dictionary()
cfg = esol_config(dictionary, embed_dim=64, n_layers=2, n_heads=8,
                  ffn_dim=128, n_kernels=16, head_hidden=32,
                  lora_targets=("q_proj", "v_proj"), r_lora=2)
weights = make_random_weights(jax.random.fold_in(key, 7), cfg)
frozen_params, params, scan_map, es_map = UniMolV1.rand_init(
    jax.random.fold_in(key, 0), weights, cfg)

print("=" * 64)
print("1. assembly + es_map audit")
print("=" * 64)
validate_assembly(params, es_map)            # raises on any es_map violation
print("validate_assembly: PASS")
theta_summary(params, es_map)

# --- 2. fake molecules (no RDKit) ---------------------------------------
def fake_mol(atoms, rng):
    coords = rng.standard_normal((len(atoms), 3)).astype(np.float32) * 1.4
    return featurize_from_coords(atoms, coords, dictionary)

rng = np.random.default_rng(0)
mols = [fake_mol(a, rng) for a in (
    ["C", "C", "O", "H", "H"], ["C", "O", "H", "H"],
    ["C", "N", "C", "O", "H", "H"], ["O", "H", "H"],
    ["C", "C", "C", "N", "O"], ["C", "F", "H", "H", "H"])]
targets = np.array([-0.7, 0.3, -1.8, 1.1, -2.4, 0.9])
scaler = TargetScaler(targets)
batch = build_batch(mols, scaler.transform(targets), dictionary, dtype=cfg.dtype)
print(f"\nbatch: {batch.src_tokens.shape[0]} molecules, "
      f"padded length {batch.src_tokens.shape[1]}")

# --- 3. single clean forward pass ---------------------------------------
print()
print("=" * 64)
print("2. forward pass (unperturbed)")
print("=" * 64)
es_tree_key = simple_es_tree_key(params, jax.random.fold_in(key, 1), scan_map)
pred0 = UniMolV1.forward(
    EggRoll, *make_noiser(params, TrainConfig(n_pop=2, pop_chunk=2)),
    frozen_params, params, es_tree_key, None,
    batch.src_tokens[0], batch.src_distance[0], batch.src_edge_type[0])
print(f"prediction for molecule 0 (standardised space): {float(pred0):+.4f}")
assert jnp.isfinite(pred0), "forward produced a non-finite value"
print("forward pass: PASS (finite scalar output)")

# --- 4. EGGROLL training loop -------------------------------------------
print()
print("=" * 64)
print("3. EGGROLL training loop (overfit the 6-molecule batch)")
print("=" * 64)
tc = TrainConfig(n_epochs=60, n_pop=16, pop_chunk=4, sigma=0.02, lr=8e-3,
                 rank=1, log_every=15, target_rmse=0.0)
frozen_noiser_params, noiser_params = make_noiser(params, tc)

# snapshot a frozen leaf (encoder layer-0 q_proj W) to prove it never moves
import jax.tree_util as jtu
def frozen_W_snapshot(p):
    return np.asarray(p["encoder"]["layers"]["0"]["attn"]["q_proj"]["W"])
W_before = frozen_W_snapshot(params)

new_params, new_noiser_params, history = train(
    EggRoll, frozen_noiser_params, noiser_params, frozen_params, params,
    es_tree_key, es_map, batch, batch,
    n_epochs=tc.n_epochs, n_pop=tc.n_pop, scaler=scaler,
    pop_chunk=tc.pop_chunk, target_rmse=tc.target_rmse, log_every=tc.log_every)

W_after = frozen_W_snapshot(new_params)

print()
print("=" * 64)
print("4. verdict")
print("=" * 64)
rmse_start = history[0]["test_rmse"]
rmse_end = history[-1]["best_test_rmse"]
frozen_ok = np.array_equal(W_before, W_after)
improved = rmse_end < rmse_start

print(f"frozen encoder weight W byte-identical before/after : {frozen_ok}")
print(f"test RMSE  start {rmse_start:.4f}  ->  best {rmse_end:.4f}  "
      f"({'decreased' if improved else 'NOT decreased'})")
ok = frozen_ok and improved and np.isfinite(rmse_end)
print()
print("SMOKE TEST:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
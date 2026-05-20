"""
01 - Main Experiment B: EGGROLL fine-tune Unimol-v1 encoder.

Pipeline:
  conformer → (perturbed) Unimol encoder → cls_repr (k=1)
                                            ↓ stack train/val
                                       ridge regression closed-form
                                            ↓
                                       fitness = -val_MSE

EGGROLL update encoder weights M[name] (subset = LoRA-targeted Linears).
Tracks best val checkpoint (early-stopping reference, not training-time stop).

Usage:
    python experiments/exp_B_encoder/01_eggroll_encoder.py \
        --cache data/cache_B \
        --out experiments/exp_B_encoder/results \
        --pop-size 64 --chunk-size 8 --rank 4 --sigma 0.01 \
        --epochs 200 --lr 1e-3 --eval-every 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eggroll import (
    EGGROLLEncoderState,
    make_vmap_forward,
    chunked_vmap_call,
    shape_fitness,
    ridge_fitness_population,
    get_unimol_encoder_lora_targets,
    print_unimol_structure,
)


# ----------------------------------------------------------------------
# Batch collation: pad list of conformer dicts into a single batch tensor
# ----------------------------------------------------------------------
def collate_conformers(conformers, padding_idx: int):
    """Pad list of dicts into batched tensors."""
    n = len(conformers)
    max_len = max(c["src_tokens"].shape[0] for c in conformers)

    src_tokens = torch.full((n, max_len), padding_idx, dtype=torch.long)
    src_distance = torch.zeros((n, max_len, max_len), dtype=torch.float32)
    src_coord = torch.zeros((n, max_len, 3), dtype=torch.float32)
    src_edge_type = torch.full((n, max_len, max_len), padding_idx, dtype=torch.long)

    for i, c in enumerate(conformers):
        L = c["src_tokens"].shape[0]
        src_tokens[i, :L] = c["src_tokens"]
        src_distance[i, :L, :L] = c["src_distance"]
        src_coord[i, :L] = c["src_coord"]
        src_edge_type[i, :L, :L] = c["src_edge_type"]

    return {
        "src_tokens": src_tokens,
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type,
    }


def make_minibatches(conformers, batch_size: int, padding_idx: int):
    """Yield collated batches (no shuffling — order preserved for ridge stacking)."""
    n = len(conformers)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield collate_conformers(conformers[start:end], padding_idx)


# ----------------------------------------------------------------------
# Forward all data through state, return cls_repr (pop, N, D)
# ----------------------------------------------------------------------
@torch.no_grad()
def forward_all_population(
    vmap_fwd, A_pop, B_pop, conformers, batch_size, padding_idx,
    chunk_size, device,
):
    """Forward ALL N molecules through encoder for each candidate.
    Returns (pop, N, embed_dim).
    """
    cls_chunks = []
    for batch in make_minibatches(conformers, batch_size, padding_idx):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["return_repr"] = True
        batch["return_atomic_reprs"] = False
        cls = chunked_vmap_call(vmap_fwd, A_pop, B_pop, batch, chunk_size=chunk_size)
        # cls: (pop, batch_size, embed_dim)
        cls_chunks.append(cls)
    return torch.cat(cls_chunks, dim=1)  # (pop, N, embed_dim)


@torch.no_grad()
def forward_all_single(model, M_overrides, conformers, batch_size, padding_idx, device):
    """Forward ALL N molecules with current M (no perturbation). Returns (N, D)."""
    import torch.func as tfunc
    cls_chunks = []
    for batch in make_minibatches(conformers, batch_size, padding_idx):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["return_repr"] = True
        batch["return_atomic_reprs"] = False
        out = tfunc.functional_call(model, M_overrides, args=(), kwargs=batch)
        cls_chunks.append(out["cls_repr"])
    return torch.cat(cls_chunks, dim=0)  # (N, D)


# ----------------------------------------------------------------------
# Compute val metrics with current M (ridge regression head)
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_current_M(
    model, state, train_conformers, val_conformers,
    y_train, y_val, batch_size, padding_idx, lam, device,
):
    """Eval current M (no perturbation). Returns dict of metrics."""
    M_overrides = {name: state.M[state.name_to_safe[name]].detach()
                   for name in state.target_param_names}
    X_train = forward_all_single(model, M_overrides, train_conformers, batch_size, padding_idx, device)
    X_val = forward_all_single(model, M_overrides, val_conformers, batch_size, padding_idx, device)

    # Ridge closed-form
    N_train, D = X_train.shape
    Xa = torch.cat([X_train, torch.ones(N_train, 1, device=device, dtype=X_train.dtype)], dim=1)
    Xav = torch.cat([X_val, torch.ones(X_val.shape[0], 1, device=device, dtype=X_val.dtype)], dim=1)
    A = Xa.T @ Xa + lam * torch.eye(D + 1, device=device, dtype=X_train.dtype)
    w = torch.linalg.solve(A, Xa.T @ y_train)
    pred_val = Xav @ w
    mse = float(torch.mean((pred_val - y_val) ** 2).item())
    rmse = float(np.sqrt(mse))
    ss_res = float(torch.sum((pred_val - y_val) ** 2).item())
    ss_tot = float(torch.sum((y_val - y_val.mean()) ** 2).item())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    pcc = float(torch.corrcoef(torch.stack([pred_val, y_val]))[0, 1].item())
    return {"mse": mse, "rmse": rmse, "r2": r2, "pearson": pcc, "ridge_lambda": lam}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/cache_B"),
                   help="parent dir; will look at {cache}/{dataset}/seed_{split_seed}/")
    p.add_argument("--dataset", type=str, default="esol")
    p.add_argument("--out", type=Path, default=Path("experiments/exp_B_encoder/results"),
                   help="results dir; will create subdir {dataset}/seed_{split_seed}/")
    p.add_argument("--data-type", type=str, default="molecule")
    p.add_argument("--remove-hs", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for EGGROLL noise (independent of split-seed)")
    p.add_argument("--split-seed", type=int, required=True,
                   help="split seed (0..4 to match Step 1)")

    # EGGROLL
    p.add_argument("--pop-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.01, help="initial sigma")
    p.add_argument("--sigma-min", type=float, default=0.001,
                   help="final sigma (cosine decay from sigma -> sigma-min)")
    p.add_argument("--sigma-decay", choices=["cosine", "none"], default="cosine")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fitness-shaping", choices=["zscore", "rank", "raw"], default="rank",
                   help="rank-based shaping is more robust to fitness outliers")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--encoder-batch-size", type=int, default=16)
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=20)

    # LoRA target selection
    p.add_argument("--targets", type=str, default="all",
                   choices=["all", "attention_only", "ffn_only"])
    p.add_argument("--layer-filter", type=str, default=None)

    p.add_argument("--print-structure", action="store_true")
    args = p.parse_args()

    out_dir = args.out / args.dataset / f"seed_{args.split_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache / args.dataset / f"seed_{args.split_seed}"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_dir}\n"
            f"Run: python experiments/exp_B_encoder/00_prepare_conformers.py "
            f"--dataset {args.dataset} --split-seed {args.split_seed}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[01] device = {device}, dataset = {args.dataset}, split_seed = {args.split_seed}")

    # ---- Load Unimol ----
    from unimol_tools.models import UniMolModel
    print(f"[01] loading UniMolModel (data_type={args.data_type}, remove_hs={args.remove_hs})")
    model = UniMolModel(output_dim=2, data_type=args.data_type, remove_hs=args.remove_hs)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    padding_idx = model.padding_idx
    print(f"     padding_idx = {padding_idx}")

    if args.print_structure:
        print_unimol_structure(model)

    # ---- LoRA targets ----
    if args.targets == "all":
        kwargs = dict(include_q=True, include_k=True, include_v=True,
                      include_out=True, include_fc1=True, include_fc2=True)
    elif args.targets == "attention_only":
        kwargs = dict(include_q=True, include_k=True, include_v=True,
                      include_out=True, include_fc1=False, include_fc2=False)
    else:
        kwargs = dict(include_q=False, include_k=False, include_v=False,
                      include_out=False, include_fc1=True, include_fc2=True)
    layer_filter = None
    if args.layer_filter:
        layer_filter = [int(x) for x in args.layer_filter.split(",")]

    targets = get_unimol_encoder_lora_targets(model, layer_filter=layer_filter, **kwargs)
    print(f"[01] LoRA targets: {len(targets)} Linear weights")
    if len(targets) == 0:
        raise RuntimeError("No LoRA targets found!")

    # ---- Load conformer cache (train + valid + test from Step 1's split) ----
    print(f"[01] loading conformer cache from {cache_dir}")
    train_data = torch.load(cache_dir / "train_conformers.pt", weights_only=False)
    valid_data = torch.load(cache_dir / "valid_conformers.pt", weights_only=False)
    test_data = torch.load(cache_dir / "test_conformers.pt", weights_only=False)
    train_conformers = train_data["conformers"]
    valid_conformers = valid_data["conformers"]
    test_conformers = test_data["conformers"]
    y_train = train_data["targets"].to(device)
    y_valid = valid_data["targets"].to(device)
    y_test = test_data["targets"].to(device)
    print(f"     train: {len(train_conformers)}, valid: {len(valid_conformers)}, test: {len(test_conformers)}")

    # ---- EGGROLL state ----
    state = EGGROLLEncoderState(
        model=model, target_param_names=targets,
        rank=args.rank, sigma=args.sigma, device=device,
    ).to(device)
    optimizer = torch.optim.Adam(state.M.parameters(), lr=args.lr)

    # ---- vmap forward ----
    vmap_fwd = make_vmap_forward(model, state, output_extractor=lambda d: d["cls_repr"])

    # ---- Training loop ----
    print(f"\n[01] training: epochs={args.epochs}, pop={args.pop_size}, chunk={args.chunk_size}, "
          f"rank={args.rank}, sigma={args.sigma}→{args.sigma_min} ({args.sigma_decay}), lr={args.lr}")
    print(f"     fitness_shaping={args.fitness_shaping}, encoder_batch={args.encoder_batch_size}, "
          f"ridge_lambda={args.ridge_lambda}")

    train_log = []
    best_val = {"rmse": float("inf"), "epoch": -1, "metrics": None, "M_snapshot": None,
                "sigma_at_best": args.sigma}
    patience_counter = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        # ---- Cosine sigma schedule ----
        if args.sigma_decay == "cosine" and args.epochs > 1:
            import math
            progress = epoch / (args.epochs - 1)
            current_sigma = args.sigma_min + 0.5 * (args.sigma - args.sigma_min) * (
                1 + math.cos(math.pi * progress)
            )
            state.sigma = current_sigma

        # Sample noise
        A_pop, B_pop = state.sample_noise(args.pop_size, use_antithetic=True)

        # Forward all train + valid through population
        t_fwd = time.time()
        cls_train_pop = forward_all_population(
            vmap_fwd, A_pop, B_pop, train_conformers,
            args.encoder_batch_size, padding_idx, args.chunk_size, device,
        )
        cls_valid_pop = forward_all_population(
            vmap_fwd, A_pop, B_pop, valid_conformers,
            args.encoder_batch_size, padding_idx, args.chunk_size, device,
        )
        fwd_t = time.time() - t_fwd

        # Ridge fitness per candidate (fit on train, eval on valid)
        fitnesses = ridge_fitness_population(
            cls_train_pop, y_train, cls_valid_pop, y_valid, lam=args.ridge_lambda,
        )

        # EGGROLL update
        shaped = shape_fitness(fitnesses, args.fitness_shaping)
        state.compute_and_assign_grads(A_pop, B_pop, shaped)
        optimizer.step()

        log_entry = {
            "epoch": epoch,
            "sigma_current": float(state.sigma),
            "fitness_mean": float(fitnesses.mean().item()),
            "fitness_max":  float(fitnesses.max().item()),
            "fitness_min":  float(fitnesses.min().item()),
            "fitness_std":  float(fitnesses.std().item()),
            "fwd_seconds":  fwd_t,
        }

        # Periodic eval on current M (valid only — test is held out)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_metrics = eval_current_M(
                model, state, train_conformers, valid_conformers,
                y_train, y_valid, args.encoder_batch_size, padding_idx,
                args.ridge_lambda, device,
            )
            log_entry["valid_metrics"] = eval_metrics
            print(f"  epoch {epoch:4d}  σ={state.sigma:.4f}  "
                  f"fit_mean={fitnesses.mean().item():+.4f} fit_max={fitnesses.max().item():+.4f}  "
                  f"valid_rmse={eval_metrics['rmse']:.4f} valid_r2={eval_metrics['r2']:.4f} "
                  f"valid_pcc={eval_metrics['pearson']:.4f}  fwd={fwd_t:.1f}s")

            if eval_metrics["rmse"] < best_val["rmse"]:
                best_val = {
                    "rmse": eval_metrics["rmse"],
                    "epoch": epoch,
                    "metrics": eval_metrics,
                    "M_snapshot": {n: state.M[state.name_to_safe[n]].detach().clone()
                                   for n in state.target_param_names},
                    "sigma_at_best": float(state.sigma),
                }
                patience_counter = 0
                print(f"        ★ new best valid RMSE = {eval_metrics['rmse']:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  early stopping at epoch {epoch} "
                          f"(patience {args.patience} exceeded; best at epoch {best_val['epoch']})")
                    train_log.append(log_entry)
                    break

        train_log.append(log_entry)

    dt = time.time() - t0
    print(f"\n[01] training done in {dt:.1f}s")

    # ---- Restore best M and evaluate on TEST set ----
    print(f"[01] restoring best M (epoch {best_val['epoch']}) and evaluating on TEST")
    for name in state.target_param_names:
        safe = state.name_to_safe[name]
        state.M[safe].data.copy_(best_val["M_snapshot"][name])

    final_train = eval_current_M(
        model, state, train_conformers, train_conformers,
        y_train, y_train, args.encoder_batch_size, padding_idx,
        args.ridge_lambda, device,
    )
    final_valid = eval_current_M(
        model, state, train_conformers, valid_conformers,
        y_train, y_valid, args.encoder_batch_size, padding_idx,
        args.ridge_lambda, device,
    )
    final_test = eval_current_M(
        model, state, train_conformers, test_conformers,
        y_train, y_test, args.encoder_batch_size, padding_idx,
        args.ridge_lambda, device,
    )

    print(f"\n[01] === FINAL METRICS (best M restored) ===")
    print(f"     Train RMSE:  {final_train['rmse']:.4f}  R²: {final_train['r2']:.4f}")
    print(f"     Valid RMSE:  {final_valid['rmse']:.4f}  R²: {final_valid['r2']:.4f}  Pearson: {final_valid['pearson']:.4f}")
    print(f"     Test  RMSE:  {final_test['rmse']:.4f}  R²: {final_test['r2']:.4f}  Pearson: {final_test['pearson']:.4f}")
    print(f"     (best at epoch {best_val['epoch']}, sigma={best_val['sigma_at_best']:.4f})")

    save = {
        "train_log": train_log,
        "best_epoch": best_val["epoch"],
        "sigma_at_best": best_val["sigma_at_best"],
        "final_train": final_train,
        "final_valid": final_valid,
        "final_test": final_test,
        "best_val_M_snapshot": best_val["M_snapshot"],
        "hyperparams": vars(args),
        "n_targets": len(targets),
        "wall_time_s": dt,
    }
    torch.save(save, out_dir / "eggroll_encoder.pt")
    (out_dir / "eggroll_encoder_summary.json").write_text(json.dumps({
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "best_epoch": best_val["epoch"],
        "train_rmse": final_train["rmse"],
        "valid_rmse": final_valid["rmse"],
        "test_rmse":  final_test["rmse"],
        "valid_r2":   final_valid["r2"],
        "test_r2":    final_test["r2"],
        "valid_pearson": final_valid["pearson"],
        "test_pearson":  final_test["pearson"],
        "n_targets": len(targets),
        "wall_time_s": dt,
        "hyperparams": {k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()},
    }, indent=2))
    print(f"[01] saved -> {out_dir}/eggroll_encoder.{{pt,_summary.json}}")


if __name__ == "__main__":
    main()
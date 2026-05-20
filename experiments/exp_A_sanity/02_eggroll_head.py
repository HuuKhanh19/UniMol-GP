"""
02 - EGGROLL: train the SAME Linear(512 -> 1) head with EGGROLL.

This is the actual sanity check: EGGROLL should converge to similar val
metrics as the gradient-descent baseline from 01_baseline_gd.py.

Setup:
  - Same model architecture, same data, same seed (where possible).
  - Population evaluated VECTORIZED (fast since model is tiny).
  - Adam used as the outer optimizer (applying EGGROLL gradient estimates).
  - Fitness = -mse on a sampled minibatch (matches GD's per-step signal).

Input:
    data/cache/esol_train.pt
    data/cache/esol_val.pt

Output:
    experiments/exp_A_sanity/results/eggroll_head.pt

Usage:
    python experiments/exp_A_sanity/02_eggroll_head.py \\
        --cache data/cache --out experiments/exp_A_sanity/results \\
        --pop-size 256 --sigma 0.05 --lr 0.01 --epochs 500
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

# Make `src` importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eggroll import EGGROLLOptimizer, linear_head_population_predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--out", type=Path, default=Path("experiments/exp_A_sanity/results"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pop-size", type=int, default=256)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fitness-shaping", choices=["zscore", "rank", "raw"], default="zscore")
    p.add_argument("--standardize-y", action="store_true",
                   help="standardize y to mean=0 std=1 (matches baseline if used)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    print(f"[02] device = {device}")

    # Load cache
    train = torch.load(args.cache / "esol_train.pt")
    val = torch.load(args.cache / "esol_val.pt")
    X_train, y_train = train["X"].to(device), train["y"].to(device)
    X_val, y_val = val["X"].to(device), val["y"].to(device)
    in_dim = X_train.shape[1]
    print(f"[02] X_train: {tuple(X_train.shape)}, X_val: {tuple(X_val.shape)}, in_dim={in_dim}")

    if args.standardize_y:
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        y_train_t = (y_train - y_mean) / y_std
    else:
        y_mean, y_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        y_train_t = y_train

    # Model — identical init pattern to baseline
    model = nn.Linear(in_dim, 1, bias=True).to(device)
    nn.init.zeros_(model.bias)
    nn.init.normal_(model.weight, std=0.02)

    # EGGROLL optimizer wraps the model's parameters; we use Adam as outer optimizer
    params = [model.weight, model.bias]
    eggroll = EGGROLLOptimizer(
        params=params,
        sigma=args.sigma,
        population_size=args.pop_size,
        use_antithetic=True,
        fitness_shaping=args.fitness_shaping,
    )
    outer_optim = torch.optim.Adam(params, lr=args.lr)

    n_train = X_train.shape[0]
    rng = np.random.default_rng(args.seed)

    print(f"[02] EGGROLL config: pop={args.pop_size}, sigma={args.sigma}, lr={args.lr}, "
          f"shaping={args.fitness_shaping}, epochs={args.epochs}, batch={args.batch_size}")

    train_curve = []
    t0 = time.time()
    for epoch in range(args.epochs):
        # 1. Sample noise for population (separate per param)
        noises = eggroll.sample_population(generator=gen)
        # noises[0]: (pop, 1, in_dim), noises[1]: (pop, 1)

        # 2. Build perturbed param batch
        W_pop = model.weight.detach().unsqueeze(0) + args.sigma * noises[0]  # (pop, 1, in_dim)
        b_pop = model.bias.detach().unsqueeze(0) + args.sigma * noises[1]    # (pop, 1)

        # 3. Sample minibatch (same minibatch for all candidates -> fair comparison)
        perm = rng.permutation(n_train)
        idx = perm[: args.batch_size]
        xb = X_train[idx]                # (B, in_dim)
        yb = y_train_t[idx]              # (B,)

        # 4. Vectorized population predict + fitness
        # preds: (pop, B)
        preds = linear_head_population_predict(xb, W_pop, b_pop)
        # fitness = -MSE per candidate; higher is better
        fitnesses = -((preds - yb.unsqueeze(0)) ** 2).mean(dim=1)  # (pop,)

        # 5. EGGROLL step (applies gradient via Adam)
        outer_optim.zero_grad(set_to_none=True)
        eggroll.step(noises, fitnesses, outer_optim)

        # 6. Evaluate val (original scale)
        with torch.no_grad():
            pred_val_t = model(X_val).squeeze(-1)
            pred_val = pred_val_t * y_std + y_mean
            val_rmse = float(torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item())
            val_r2 = 1.0 - float(torch.sum((pred_val - y_val) ** 2)) / max(
                float(torch.sum((y_val - y_val.mean()) ** 2)), 1e-12
            )
            val_pearson = float(torch.corrcoef(torch.stack([pred_val, y_val]))[0, 1].item())

        train_curve.append({
            "epoch": epoch,
            "train_fitness_mean": float(fitnesses.mean().item()),
            "train_fitness_max": float(fitnesses.max().item()),
            "train_fitness_min": float(fitnesses.min().item()),
            "train_fitness_std": float(fitnesses.std().item()),
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "val_pearson": val_pearson,
        })

        if epoch % max(args.epochs // 20, 1) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  fitness(mean/max)={fitnesses.mean().item():.4f}/"
                  f"{fitnesses.max().item():.4f}  "
                  f"val_rmse={val_rmse:.4f}  val_r2={val_r2:.4f}  val_pearson={val_pearson:.4f}")

    dt = time.time() - t0
    print(f"[02] done in {dt:.1f}s")

    final = {
        "rmse": train_curve[-1]["val_rmse"],
        "r2": train_curve[-1]["val_r2"],
        "pearson": train_curve[-1]["val_pearson"],
    }
    print(f"[02] FINAL val: rmse={final['rmse']:.4f}  r2={final['r2']:.4f}  pearson={final['pearson']:.4f}")

    save_path = args.out / "eggroll_head.pt"
    torch.save({
        "model_state": model.state_dict(),
        "train_curve": train_curve,
        "final_val": final,
        "hyperparams": vars(args),
        "wall_time_s": dt,
    }, save_path)
    print(f"[02] saved -> {save_path}")


if __name__ == "__main__":
    main()
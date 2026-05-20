"""
01 - Baseline: train a Linear(512 -> 1) head with gradient descent (Adam).

This is the reference EGGROLL must match in Experiment A.

Input:
    data/cache/esol_train.pt
    data/cache/esol_val.pt

Output:
    experiments/exp_A_sanity/results/baseline_gd.pt   {train_curve, val_curve, final_weights, ...}

Usage:
    python experiments/exp_A_sanity/01_baseline_gd.py \\
        --cache data/cache --out experiments/exp_A_sanity/results
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def evaluate(model: nn.Linear, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(X).squeeze(-1)
        mse = torch.mean((pred - y) ** 2).item()
        rmse = float(np.sqrt(mse))
        # R²
        ss_res = torch.sum((pred - y) ** 2).item()
        ss_tot = torch.sum((y - y.mean()) ** 2).item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        # Pearson
        pcc = float(torch.corrcoef(torch.stack([pred, y]))[0, 1].item())
    return {"mse": mse, "rmse": rmse, "r2": r2, "pearson": pcc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--out", type=Path, default=Path("experiments/exp_A_sanity/results"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64,
                   help="set to -1 (or >= N_train) for full-batch GD; "
                        "full-batch should converge to closed-form ridge "
                        "(SGD with small batch beats closed-form via implicit regularization)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standardize-y", action="store_true",
                   help="standardize y to mean=0 std=1 during training; metrics reported on original scale")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[01] device = {device}")

    # Load cache
    train = torch.load(args.cache / "esol_train.pt")
    val = torch.load(args.cache / "esol_val.pt")
    X_train, y_train = train["X"].to(device), train["y"].to(device)
    X_val, y_val = val["X"].to(device), val["y"].to(device)
    in_dim = X_train.shape[1]
    print(f"[01] X_train: {tuple(X_train.shape)}, X_val: {tuple(X_val.shape)}, in_dim={in_dim}")

    # Optionally standardize y
    if args.standardize_y:
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        y_train_t = (y_train - y_mean) / y_std
        y_val_t = (y_val - y_mean) / y_std
    else:
        y_mean, y_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        y_train_t, y_val_t = y_train, y_val

    # Model: single Linear(512 -> 1)
    model = nn.Linear(in_dim, 1, bias=True).to(device)
    nn.init.zeros_(model.bias)
    nn.init.normal_(model.weight, std=0.02)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    rng = np.random.default_rng(args.seed)

    # Handle full-batch mode
    effective_batch = args.batch_size if args.batch_size > 0 else n_train
    effective_batch = min(effective_batch, n_train)
    is_full_batch = effective_batch >= n_train
    mode_str = "FULL-BATCH GD" if is_full_batch else f"SGD batch={effective_batch}"

    train_curve = []  # list of dicts: {epoch, train_loss, val_metrics_dict}
    print(f"[01] training for {args.epochs} epochs, mode={mode_str}, lr={args.lr}")

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n_train)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, effective_batch):
            idx = perm[start : start + effective_batch]
            xb = X_train[idx]
            yb = y_train_t[idx]
            optim.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Evaluate on val (on original scale)
        model.eval()
        with torch.no_grad():
            pred_val_t = model(X_val).squeeze(-1)
            pred_val = pred_val_t * y_std + y_mean  # back to original scale
        val_metrics = {
            "mse": float(torch.mean((pred_val - y_val) ** 2).item()),
            "rmse": float(torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item()),
            "r2": 1.0 - float(torch.sum((pred_val - y_val) ** 2)) / max(float(torch.sum((y_val - y_val.mean()) ** 2)), 1e-12),
            "pearson": float(torch.corrcoef(torch.stack([pred_val, y_val]))[0, 1].item()),
        }
        train_curve.append({
            "epoch": epoch,
            "train_loss_standardized": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if epoch % max(args.epochs // 20, 1) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  train_loss={train_loss:.5f}  "
                  f"val_rmse={val_metrics['rmse']:.4f}  val_r2={val_metrics['r2']:.4f}  "
                  f"val_pearson={val_metrics['pearson']:.4f}")

    dt = time.time() - t0
    print(f"[01] done in {dt:.1f}s")

    # Final metrics
    final = evaluate(model, X_val, y_val) if not args.standardize_y else {
        # already in train_curve[-1]
        k.replace("val_", ""): v for k, v in train_curve[-1].items() if k.startswith("val_")
    }
    print(f"[01] FINAL val: rmse={final['rmse']:.4f}  r2={final['r2']:.4f}  pearson={final['pearson']:.4f}")

    # Also compute closed-form ridge for reference (lambda=0 = OLS)
    with torch.no_grad():
        Xa = torch.cat([X_train, torch.ones(X_train.shape[0], 1, device=device)], dim=1)  # (N, 513)
        # ridge with very small lambda for stability
        lam = 1e-4
        A = Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=device)
        w = torch.linalg.solve(A, Xa.T @ y_train)
        Xva = torch.cat([X_val, torch.ones(X_val.shape[0], 1, device=device)], dim=1)
        pred_cf = Xva @ w
        cf_rmse = float(torch.sqrt(torch.mean((pred_cf - y_val) ** 2)).item())
        cf_r2 = 1.0 - float(torch.sum((pred_cf - y_val) ** 2)) / max(float(torch.sum((y_val - y_val.mean()) ** 2)), 1e-12)
        cf_pcc = float(torch.corrcoef(torch.stack([pred_cf, y_val]))[0, 1].item())
    print(f"[01] CLOSED-FORM ridge (lambda={lam}): rmse={cf_rmse:.4f}  r2={cf_r2:.4f}  pearson={cf_pcc:.4f}")
    print(f"     (this is the analytical optimum; GD/EGGROLL should get close to it)")

    # Save
    save_path = args.out / "baseline_gd.pt"
    torch.save({
        "model_state": model.state_dict(),
        "train_curve": train_curve,
        "final_val": final,
        "closed_form_val": {"rmse": cf_rmse, "r2": cf_r2, "pearson": cf_pcc, "lambda": lam},
        "hyperparams": vars(args),
        "wall_time_s": dt,
    }, save_path)
    print(f"[01] saved -> {save_path}")


if __name__ == "__main__":
    main()
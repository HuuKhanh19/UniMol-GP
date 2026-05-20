"""Smoke test: chạy EGGROLL trên synthetic linear data, không cần Unimol.
Verify rằng pipeline EGGROLL hoạt động đúng trước khi user chạy với data thật.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from src.eggroll import EGGROLLOptimizer, linear_head_population_predict


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cpu"

    # Synthetic linear-regression task: y = X @ w_true + noise
    N_train, N_val, D = 800, 200, 64
    w_true = torch.randn(D)
    b_true = torch.tensor(0.5)
    X_train = torch.randn(N_train, D)
    y_train = X_train @ w_true + b_true + 0.05 * torch.randn(N_train)
    X_val = torch.randn(N_val, D)
    y_val = X_val @ w_true + b_true + 0.05 * torch.randn(N_val)

    # ---- Closed-form OLS reference ----
    Xa = torch.cat([X_train, torch.ones(N_train, 1)], dim=1)
    Xva = torch.cat([X_val, torch.ones(N_val, 1)], dim=1)
    w_cf = torch.linalg.solve(Xa.T @ Xa + 1e-4 * torch.eye(D + 1), Xa.T @ y_train)
    pred_cf = Xva @ w_cf
    cf_rmse = torch.sqrt(torch.mean((pred_cf - y_val) ** 2)).item()
    print(f"[CF OLS]  val RMSE = {cf_rmse:.4f}")

    # ---- GD baseline (give it enough steps to converge for fair comparison) ----
    model_gd = nn.Linear(D, 1)
    nn.init.zeros_(model_gd.bias); nn.init.normal_(model_gd.weight, std=0.02)
    opt = torch.optim.Adam(model_gd.parameters(), lr=3e-2)
    for epoch in range(2000):
        idx = torch.randperm(N_train)[:64]
        pred = model_gd(X_train[idx]).squeeze(-1)
        loss = ((pred - y_train[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    gd_rmse = torch.sqrt(torch.mean((model_gd(X_val).squeeze(-1) - y_val) ** 2)).item()
    print(f"[GD]      val RMSE = {gd_rmse:.4f}")

    # ---- EGGROLL ----
    model_eg = nn.Linear(D, 1)
    nn.init.zeros_(model_eg.bias); nn.init.normal_(model_eg.weight, std=0.02)
    eggroll = EGGROLLOptimizer(
        params=[model_eg.weight, model_eg.bias],
        sigma=0.05, population_size=256,
        use_antithetic=True, fitness_shaping="zscore",
    )
    outer = torch.optim.Adam([model_eg.weight, model_eg.bias], lr=0.01)
    for epoch in range(2000):
        noises = eggroll.sample_population()
        W_pop = model_eg.weight.detach().unsqueeze(0) + 0.05 * noises[0]
        b_pop = model_eg.bias.detach().unsqueeze(0) + 0.05 * noises[1]
        idx = torch.randperm(N_train)[:64]
        xb, yb = X_train[idx], y_train[idx]
        preds = linear_head_population_predict(xb, W_pop, b_pop)
        fitnesses = -((preds - yb.unsqueeze(0)) ** 2).mean(dim=1)
        outer.zero_grad()
        eggroll.step(noises, fitnesses, outer)
        if epoch % 250 == 0:
            with torch.no_grad():
                val_rmse = torch.sqrt(torch.mean((model_eg(X_val).squeeze(-1) - y_val) ** 2)).item()
            print(f"  EGGROLL epoch {epoch:4d}: val_rmse={val_rmse:.4f}, "
                  f"fitness mean={fitnesses.mean().item():.4f}")
    eg_rmse = torch.sqrt(torch.mean((model_eg(X_val).squeeze(-1) - y_val) ** 2)).item()
    print(f"[EGGROLL] val RMSE = {eg_rmse:.4f}")

    # Verdict: both methods should be reasonably close to closed-form
    gd_gap_to_cf = (gd_rmse - cf_rmse) / cf_rmse
    eg_gap_to_cf = (eg_rmse - cf_rmse) / cf_rmse
    print(f"\nGD      vs closed-form gap = {gd_gap_to_cf*100:+.2f}%")
    print(f"EGGROLL vs closed-form gap = {eg_gap_to_cf*100:+.2f}%")
    if gd_gap_to_cf < 0.50 and eg_gap_to_cf < 0.50:
        print("✅ Smoke test PASS — both methods converge near optimum, implementation correct.")
    elif eg_gap_to_cf < 1.0:
        print("⚠️  EGGROLL converging but slowly; implementation likely correct, may need more steps.")
    else:
        print("❌ Smoke test FAIL — investigate EGGROLL implementation.")


if __name__ == "__main__":
    main()
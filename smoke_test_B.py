"""
Smoke test for Experiment B core mechanics (vmap + functional_call + ridge fitness).

Uses a TOY encoder (1 Linear) so we don't need Unimol — just verifies:
  1. EGGROLLEncoderState builds correctly
  2. vmap over (A, B) candidates works with functional_call
  3. chunked_vmap_call concatenates chunks correctly
  4. ridge_fitness_population matches manual computation
  5. compute_and_assign_grads + Adam.step() updates M

If this passes, the core vmap pipeline is correct.
After this, plug in Unimol (which is just a more complex model).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from src.eggroll import (
    EGGROLLEncoderState,
    make_vmap_forward,
    chunked_vmap_call,
    shape_fitness,
    ridge_fitness_population,
)


class ToyEncoder(nn.Module):
    """Tiny 'encoder' = single Linear, to test the mechanics."""
    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.encoder = nn.Sequential()
        self.encoder.layers = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False)])

    def forward(self, x):
        # Returns dict like Unimol's return_repr=True format
        h = self.encoder.layers[0](x)
        return {"cls_repr": h.mean(dim=1) if h.dim() == 3 else h}  # (B, out)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ---- Synthetic data ----
    # Task: predict y = some non-linear function of x mean
    in_dim, embed_dim = 8, 4
    N_train, N_val = 100, 30
    X_train = torch.randn(N_train, 3, in_dim, device=device)  # (B, seq=3, in_dim)
    X_val = torch.randn(N_val, 3, in_dim, device=device)
    # True target: function of x averaged
    w_true = torch.randn(in_dim, device=device)
    y_train = (X_train.mean(dim=1) @ w_true) + 0.1 * torch.randn(N_train, device=device)
    y_val = (X_val.mean(dim=1) @ w_true) + 0.1 * torch.randn(N_val, device=device)

    # ---- Toy encoder ----
    model = ToyEncoder(in_dim=in_dim, out_dim=embed_dim).to(device)
    target_names = ["encoder.layers.0.weight"]

    print(f"\nTarget params: {target_names}")
    print(f"  Shape: {dict(model.named_parameters())[target_names[0]].shape}")

    # ---- EGGROLL state ----
    state = EGGROLLEncoderState(
        model=model,
        target_param_names=target_names,
        rank=2,
        sigma=0.05,
        device=device,
    ).to(device)

    pop = 16
    A_pop, B_pop = state.sample_noise(pop, use_antithetic=True)
    print(f"\nA_pop[name].shape = {A_pop[target_names[0]].shape}  (expect ({pop}, {embed_dim}, 2))")
    print(f"B_pop[name].shape = {B_pop[target_names[0]].shape}  (expect ({pop}, 2, {in_dim}))")

    # ---- Vmap forward ----
    vmap_fwd = make_vmap_forward(model, state, output_extractor=lambda d: d["cls_repr"])

    # Single chunk test
    cls_train_pop = vmap_fwd(A_pop, B_pop, x=X_train)
    print(f"\nSingle vmap call: cls_repr.shape = {tuple(cls_train_pop.shape)}  "
          f"(expect ({pop}, {N_train}, {embed_dim}))")
    assert cls_train_pop.shape == (pop, N_train, embed_dim), "shape mismatch!"

    # Chunked call test (chunk_size 4)
    cls_train_chunked = chunked_vmap_call(vmap_fwd, A_pop, B_pop, {"x": X_train}, chunk_size=4)
    print(f"Chunked vmap (chunk=4): max abs diff vs single = "
          f"{(cls_train_chunked - cls_train_pop).abs().max().item():.2e}")
    assert torch.allclose(cls_train_chunked, cls_train_pop, atol=1e-5), "chunk concat mismatch!"

    # ---- Ridge fitness ----
    cls_val_pop = vmap_fwd(A_pop, B_pop, x=X_val)
    fitnesses = ridge_fitness_population(
        cls_train_pop, y_train, cls_val_pop, y_val, lam=1e-3
    )
    print(f"\nFitness shape = {tuple(fitnesses.shape)}  (expect ({pop},))")
    print(f"Fitness range: [{fitnesses.min().item():.4f}, {fitnesses.max().item():.4f}]")
    print(f"  mean = {fitnesses.mean().item():.4f}")

    # Manual check on candidate 0
    cls_t0 = cls_train_pop[0]  # (N_train, embed_dim)
    cls_v0 = cls_val_pop[0]
    Xa_t = torch.cat([cls_t0, torch.ones(N_train, 1, device=device)], dim=1)
    Xa_v = torch.cat([cls_v0, torch.ones(N_val, 1, device=device)], dim=1)
    w_manual = torch.linalg.solve(
        Xa_t.T @ Xa_t + 1e-3 * torch.eye(embed_dim + 1, device=device),
        Xa_t.T @ y_train
    )
    pred = Xa_v @ w_manual
    mse_manual = ((pred - y_val) ** 2).mean().item()
    fitness_manual = -mse_manual
    print(f"  manual candidate-0 fitness: {fitness_manual:.4f} "
          f"(vmap: {fitnesses[0].item():.4f})")
    assert abs(fitness_manual - fitnesses[0].item()) < 1e-3, "ridge fitness mismatch!"

    # ---- Training loop test ----
    optim = torch.optim.Adam(state.M.parameters(), lr=0.01)
    init_M = state.M[target_names[0].replace(".", "__")].detach().clone()

    print("\nTraining loop (50 steps):")
    for step in range(50):
        A_pop, B_pop = state.sample_noise(pop, use_antithetic=True)
        cls_t = chunked_vmap_call(vmap_fwd, A_pop, B_pop, {"x": X_train}, chunk_size=8)
        cls_v = chunked_vmap_call(vmap_fwd, A_pop, B_pop, {"x": X_val}, chunk_size=8)
        fits = ridge_fitness_population(cls_t, y_train, cls_v, y_val, lam=1e-3)
        shaped = shape_fitness(fits, "zscore")
        state.compute_and_assign_grads(A_pop, B_pop, shaped)
        optim.step()
        if step % 10 == 0 or step == 49:
            # Evaluate current M (no perturbation)
            zero_A = {n: torch.zeros_like(A_pop[n][:1]) for n in target_names}
            zero_B = {n: torch.zeros_like(B_pop[n][:1]) for n in target_names}
            cls_t_eval = vmap_fwd(zero_A, zero_B, x=X_train)
            cls_v_eval = vmap_fwd(zero_A, zero_B, x=X_val)
            eval_fit = ridge_fitness_population(cls_t_eval, y_train, cls_v_eval, y_val, lam=1e-3)
            print(f"  step {step:3d}  fit_mean={fits.mean().item():+.4f}  "
                  f"M_eval_fit={eval_fit.item():+.4f}  "
                  f"||delta_M||={torch.norm(state.M[target_names[0].replace('.','__')] - init_M).item():.4f}")

    print("\n✅ Smoke test B PASS — vmap pipeline is correct.")


if __name__ == "__main__":
    main()
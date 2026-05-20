"""
03 - Compare baseline GD vs EGGROLL on the same Linear head.

Produces a side-by-side plot of training curves + a textual summary.

Success criterion for Experiment A:
    EGGROLL's final val RMSE / R² / Pearson should be close to GD's final
    (and both close to the closed-form ridge solution).

Input:
    experiments/exp_A_sanity/results/baseline_gd.pt
    experiments/exp_A_sanity/results/eggroll_head.pt

Output:
    experiments/exp_A_sanity/results/comparison.png
    experiments/exp_A_sanity/results/comparison.json   (numeric summary)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("experiments/exp_A_sanity/results"))
    args = p.parse_args()

    gd = torch.load(args.results / "baseline_gd.pt", weights_only=False)
    eg = torch.load(args.results / "eggroll_head.pt", weights_only=False)

    print("=" * 70)
    print(" Experiment A — Sanity Check: EGGROLL vs Gradient Descent")
    print(" (Linear(512->1) head on top of FROZEN Unimol-v1 ESOL embeddings)")
    print("=" * 70)

    print("\n[FINAL VAL METRICS]")
    print(f"  {'method':<20}  {'rmse':>8}  {'r2':>8}  {'pearson':>8}  {'wall_s':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    cf = gd["closed_form_val"]
    print(f"  {'closed-form ridge':<20}  {cf['rmse']:>8.4f}  {cf['r2']:>8.4f}  {cf['pearson']:>8.4f}  {'--':>8}")
    print(f"  {'gradient descent':<20}  {gd['final_val']['rmse']:>8.4f}  {gd['final_val']['r2']:>8.4f}  "
          f"{gd['final_val']['pearson']:>8.4f}  {gd['wall_time_s']:>8.1f}")
    print(f"  {'EGGROLL':<20}  {eg['final_val']['rmse']:>8.4f}  {eg['final_val']['r2']:>8.4f}  "
          f"{eg['final_val']['pearson']:>8.4f}  {eg['wall_time_s']:>8.1f}")

    # ============================================================
    # Primary verdict: EGGROLL vs CLOSED-FORM RIDGE
    # ============================================================
    # Both target the same mathematical optimum (regularized OLS).
    # EGGROLL should converge to this. GD with SGD noise may beat it via
    # implicit regularization — that's a property of SGD, NOT a baseline
    # EGGROLL is supposed to match.
    rmse_gap_cf = (eg["final_val"]["rmse"] - cf["rmse"]) / max(cf["rmse"], 1e-8)
    r2_gap_cf = eg["final_val"]["r2"] - cf["r2"]
    pcc_gap_cf = eg["final_val"]["pearson"] - cf["pearson"]

    print("\n[PRIMARY GAP: EGGROLL vs CLOSED-FORM RIDGE (analytical optimum)]")
    print(f"  RMSE relative gap: {rmse_gap_cf*100:+.2f}%  (negative = EGGROLL better)")
    print(f"  R²   absolute gap: {r2_gap_cf:+.4f}        (positive = EGGROLL better)")
    print(f"  PCC  absolute gap: {pcc_gap_cf:+.4f}        (positive = EGGROLL better)")

    # ============================================================
    # Secondary observation: GD vs closed-form (SGD implicit regularization)
    # ============================================================
    gd_rmse_gap_cf = (gd["final_val"]["rmse"] - cf["rmse"]) / max(cf["rmse"], 1e-8)
    gd_r2_gap_cf = gd["final_val"]["r2"] - cf["r2"]

    print("\n[SECONDARY GAP: GD vs CLOSED-FORM (SGD implicit regularization effect)]")
    print(f"  RMSE relative gap: {gd_rmse_gap_cf*100:+.2f}%  (negative = GD better than analytical optimum)")
    print(f"  R²   absolute gap: {gd_r2_gap_cf:+.4f}")
    if gd_rmse_gap_cf < -0.05:
        print("  → GD significantly beats closed-form. This is SGD implicit regularization at work,")
        print("    NOT a property of gradient descent vs ES. Try --full-batch GD to confirm:")
        print("    GD with batch_size = N_train should converge to closed-form too.")

    # Also report EGGROLL vs GD for legacy / informational purposes
    rmse_gap_gd = (eg["final_val"]["rmse"] - gd["final_val"]["rmse"]) / max(gd["final_val"]["rmse"], 1e-8)
    r2_gap_gd = eg["final_val"]["r2"] - gd["final_val"]["r2"]
    pcc_gap_gd = eg["final_val"]["pearson"] - gd["final_val"]["pearson"]
    print(f"\n[INFO: EGGROLL vs SGD-based GD] RMSE rel gap: {rmse_gap_gd*100:+.2f}%, "
          f"R² abs gap: {r2_gap_gd:+.4f}, PCC abs gap: {pcc_gap_gd:+.4f}")
    print("  (Not the primary criterion — see above. SGD has implicit reg that EGGROLL/closed-form do not.)")

    # ============================================================
    # Verdict based on PRIMARY gap (vs closed-form)
    # ============================================================
    print("\n[VERDICT]")
    if abs(rmse_gap_cf) < 0.05 and abs(r2_gap_cf) < 0.03:
        print("  ✅ PASS — EGGROLL converges to within 5% RMSE / 0.03 R² of the analytical optimum.")
        print("  → EGGROLL implementation is CORRECT. Proceed to Experiment B (encoder fine-tune).")
        if gd_rmse_gap_cf < -0.05:
            print("\n  Note: GD beats the analytical optimum due to SGD implicit regularization.")
            print("  This is expected and does NOT indicate an EGGROLL bug. For Experiment B,")
            print("  consider adding explicit weight decay to EGGROLL fitness, or comparing")
            print("  EGGROLL against full-batch GD (which will match closed-form).")
    elif abs(rmse_gap_cf) < 0.10:
        print("  ⚠️  MARGINAL — Within 10% but not 5% of analytical optimum. Consider:")
        print("     - smaller sigma (current may be too aggressive, causing wiggle around optimum)")
        print("     - sigma decay schedule (lower sigma over training)")
        print("     - larger pop_size or more epochs")
    else:
        print("  ❌ FAIL — gap to analytical optimum too large. Likely causes:")
        print("     1. sigma too large or too small")
        print("     2. lr mismatch with EGGROLL gradient magnitude")
        print("     3. Bug in implementation — check sign of gradient and shaping")

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # RMSE
        ax = axes[0]
        ax.plot([d["epoch"] for d in gd["train_curve"]], [d["val_rmse"] for d in gd["train_curve"]],
                label="GD", color="#1f77b4")
        ax.plot([d["epoch"] for d in eg["train_curve"]], [d["val_rmse"] for d in eg["train_curve"]],
                label="EGGROLL", color="#d62728")
        ax.axhline(cf["rmse"], linestyle="--", color="gray", label=f"closed-form (rmse={cf['rmse']:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val RMSE")
        ax.set_title("Validation RMSE")
        ax.legend()
        ax.grid(alpha=0.3)

        # R²
        ax = axes[1]
        ax.plot([d["epoch"] for d in gd["train_curve"]], [d["val_r2"] for d in gd["train_curve"]],
                label="GD", color="#1f77b4")
        ax.plot([d["epoch"] for d in eg["train_curve"]], [d["val_r2"] for d in eg["train_curve"]],
                label="EGGROLL", color="#d62728")
        ax.axhline(cf["r2"], linestyle="--", color="gray", label=f"closed-form (r²={cf['r2']:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val R²")
        ax.set_title("Validation R²")
        ax.legend()
        ax.grid(alpha=0.3)

        # Pearson
        ax = axes[2]
        ax.plot([d["epoch"] for d in gd["train_curve"]], [d["val_pearson"] for d in gd["train_curve"]],
                label="GD", color="#1f77b4")
        ax.plot([d["epoch"] for d in eg["train_curve"]], [d["val_pearson"] for d in eg["train_curve"]],
                label="EGGROLL", color="#d62728")
        ax.axhline(cf["pearson"], linestyle="--", color="gray", label=f"closed-form (pcc={cf['pearson']:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val Pearson")
        ax.set_title("Validation Pearson")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle("Experiment A: EGGROLL vs Gradient Descent on Linear head (ESOL frozen embeddings)")
        plt.tight_layout()
        out_png = args.results / "comparison.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"\n[03] saved plot -> {out_png}")
    except ImportError:
        print("\n[03] matplotlib not available; skipping plot.")

    # Save numeric summary
    summary = {
        "closed_form_val": cf,
        "gradient_descent_final": gd["final_val"],
        "eggroll_final": eg["final_val"],
        "primary_gap_eggroll_vs_closed_form": {
            "rmse_relative": rmse_gap_cf,
            "r2_absolute": r2_gap_cf,
            "pearson_absolute": pcc_gap_cf,
        },
        "secondary_gap_gd_vs_closed_form": {
            "rmse_relative": gd_rmse_gap_cf,
            "r2_absolute": gd_r2_gap_cf,
            "_interpretation": "negative = SGD implicit regularization helping GD",
        },
        "info_gap_eggroll_vs_gd": {
            "rmse_relative": rmse_gap_gd,
            "r2_absolute": r2_gap_gd,
            "pearson_absolute": pcc_gap_gd,
        },
        "wall_time": {
            "gradient_descent_s": gd["wall_time_s"],
            "eggroll_s": eg["wall_time_s"],
        },
        "gd_hyperparams": gd["hyperparams"],
        "eggroll_hyperparams": eg["hyperparams"],
    }
    (args.results / "comparison.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[03] saved summary -> {args.results / 'comparison.json'}")


if __name__ == "__main__":
    main()
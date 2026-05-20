"""
02 - Aggregate EGGROLL multi-seed results & compare with Step 1 baseline.

Reads:
    experiments/exp_B_encoder/results/seed_{0..4}/eggroll_encoder_summary.json

Prints:
    Table: per-seed metrics + mean ± std for both Step 1 and EGGROLL.

Step 1 baseline numbers are hardcoded from user's provided output.
Update STEP1_RESULTS dict if running on different dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


# Step 1 baseline from user's MolTrain runs (refined_ESOL, 5 split-seeds)
STEP1_RESULTS = {
    0: {"train_rmse": 0.3304, "valid_rmse": 0.7241, "test_rmse": 0.8579},
    1: {"train_rmse": 0.3541, "valid_rmse": 0.7569, "test_rmse": 0.8368},
    2: {"train_rmse": 0.2460, "valid_rmse": 0.6591, "test_rmse": 0.7579},
    3: {"train_rmse": 0.3315, "valid_rmse": 0.6718, "test_rmse": 0.8630},
    4: {"train_rmse": 0.2459, "valid_rmse": 0.6311, "test_rmse": 0.8376},
}


def mean_std(values):
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("experiments/exp_B_encoder/results"),
                   help="dir containing {dataset}/seed_{0..4}/eggroll_encoder_summary.json")
    p.add_argument("--dataset", type=str, default="esol")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="comma-separated split seeds to aggregate")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    # Load EGGROLL results
    eggroll = {}
    missing = []
    for s in seeds:
        path = args.results / args.dataset / f"seed_{s}" / "eggroll_encoder_summary.json"
        if not path.exists():
            missing.append(s)
            continue
        eggroll[s] = json.loads(path.read_text())

    if missing:
        print(f"⚠️  Missing EGGROLL results for seeds: {missing}")
        if not eggroll:
            print(f"   No EGGROLL results found under {args.results}/{args.dataset}/")
            return

    available_seeds = sorted(eggroll.keys())

    print("=" * 80)
    print(f" EGGROLL Experiment B vs Step 1 baseline ({len(available_seeds)} seeds)")
    print("=" * 80)

    # Per-seed table
    print(f"\n{'seed':<6}{'method':<12}{'train':>10}{'valid':>10}{'test':>10}{'best_ep':>10}{'wall(s)':>10}")
    print("-" * 68)
    for s in available_seeds:
        st = STEP1_RESULTS.get(s, None)
        eg = eggroll[s]
        if st:
            print(f"{s:<6}{'Step 1':<12}{st['train_rmse']:>10.4f}{st['valid_rmse']:>10.4f}"
                  f"{st['test_rmse']:>10.4f}{'--':>10}{'--':>10}")
        print(f"{s:<6}{'EGGROLL':<12}{eg['train_rmse']:>10.4f}{eg['valid_rmse']:>10.4f}"
              f"{eg['test_rmse']:>10.4f}{eg['best_epoch']:>10d}{eg['wall_time_s']:>10.0f}")
        print()

    # Aggregate
    step1_seeds = [s for s in available_seeds if s in STEP1_RESULTS]
    print("=" * 80)
    print(f"\n{'method':<12}{'metric':<10}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
    print("-" * 62)

    for metric in ["train_rmse", "valid_rmse", "test_rmse"]:
        if step1_seeds:
            vals = [STEP1_RESULTS[s][metric] for s in step1_seeds]
            m, sd = mean_std(vals)
            print(f"{'Step 1':<12}{metric:<10}{m:>10.4f}{sd:>10.4f}{min(vals):>10.4f}{max(vals):>10.4f}")

        vals = [eggroll[s][metric] for s in available_seeds]
        m, sd = mean_std(vals)
        print(f"{'EGGROLL':<12}{metric:<10}{m:>10.4f}{sd:>10.4f}{min(vals):>10.4f}{max(vals):>10.4f}")
        print()

    # Verdict (based on test RMSE, which is what matters)
    if step1_seeds:
        st1_test = [STEP1_RESULTS[s]["test_rmse"] for s in step1_seeds]
        eg_test = [eggroll[s]["test_rmse"] for s in available_seeds]
        m_st1, _ = mean_std(st1_test)
        m_eg, _ = mean_std(eg_test)
        gap_pct = (m_eg - m_st1) / m_st1 * 100

        print("=" * 80)
        print(f" VERDICT (test RMSE):")
        print(f"   Step 1 mean:  {m_st1:.4f}")
        print(f"   EGGROLL mean: {m_eg:.4f}")
        print(f"   Gap: {gap_pct:+.2f}%  (negative = EGGROLL better)")
        if gap_pct < -2:
            print(f"   🏆 EGGROLL BEATS Step 1 by {-gap_pct:.1f}% on test RMSE!")
        elif gap_pct < 5:
            print(f"   ≈ EGGROLL matches Step 1 (within 5%).")
        elif gap_pct < 15:
            print(f"   ⚠️  EGGROLL behind by {gap_pct:.1f}% — tune hyperparams further.")
        else:
            print(f"   ❌ Gap large ({gap_pct:.1f}%) — significant tuning needed.")

    # Save aggregated JSON
    out_path = args.results / args.dataset / "aggregated_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "step1_per_seed": {s: STEP1_RESULTS.get(s) for s in available_seeds if s in STEP1_RESULTS},
        "eggroll_per_seed": {s: eggroll[s] for s in available_seeds},
        "aggregates": {
            "step1": {
                metric: dict(zip(["mean", "std"], mean_std([STEP1_RESULTS[s][metric] for s in step1_seeds])))
                for metric in ["train_rmse", "valid_rmse", "test_rmse"]
            } if step1_seeds else None,
            "eggroll": {
                metric: dict(zip(["mean", "std"], mean_std([eggroll[s][metric] for s in available_seeds])))
                for metric in ["train_rmse", "valid_rmse", "test_rmse"]
            },
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[02] saved aggregated summary -> {out_path}")


if __name__ == "__main__":
    main()
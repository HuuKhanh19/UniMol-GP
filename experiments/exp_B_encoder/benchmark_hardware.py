"""
Benchmark script: find max (chunk-size, encoder-batch-size) for RTX 5070 Ti 16GB.

Tests pairs of (chunk_size, batch_size), runs 3 epochs each, measures:
  - Average epoch wall-clock
  - Peak GPU memory used

These params DO NOT affect EGGROLL training quality — only speed/memory.
After finding the max safe config, you can use it for all future runs to go faster.

Usage:
    python experiments/exp_B_encoder/benchmark_hardware.py `
        --dataset esol --split-seed 0 `
        --pop-size 64 --rank 4

Will print a table like:
    chunk  batch  peak_mem_GB  epoch_s  status
        4      8         3.2     65.4   OK
        4     16         3.5     41.2   OK
        8     16         4.6     31.5   OK   ← current default
       16     16         6.7     22.1   OK
       16     32         7.2     18.4   OK
       32     16        10.2     17.3   OK
       32     32        11.0     14.2   OK
       64     16        OOM      -      OOM
       64     32        OOM      -      OOM
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eggroll import (
    EGGROLLEncoderState,
    make_vmap_forward,
    chunked_vmap_call,
    shape_fitness,
    ridge_fitness_population,
    get_unimol_encoder_lora_targets,
)

# Import helpers from the main training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_eg", str(Path(__file__).resolve().parent / "01_eggroll_encoder.py")
)
_eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eg)
make_minibatches = _eg.make_minibatches
forward_all_population = _eg.forward_all_population


def benchmark_one_config(
    model, state, vmap_fwd, padding_idx,
    train_conformers, valid_conformers, y_train, y_valid,
    chunk_size: int, encoder_batch_size: int, pop_size: int,
    n_epochs: int = 3, device: torch.device = None,
):
    """Run n_epochs of training with given (chunk, batch). Return (peak_mem_GB, avg_epoch_s)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    epoch_times = []
    for epoch in range(n_epochs):
        A_pop, B_pop = state.sample_noise(pop_size, use_antithetic=True)

        t0 = time.time()
        cls_train_pop = forward_all_population(
            vmap_fwd, A_pop, B_pop, train_conformers,
            encoder_batch_size, padding_idx, chunk_size, device,
        )
        cls_valid_pop = forward_all_population(
            vmap_fwd, A_pop, B_pop, valid_conformers,
            encoder_batch_size, padding_idx, chunk_size, device,
        )
        # Ridge fitness (small compute compared to forward)
        fitnesses = ridge_fitness_population(
            cls_train_pop, y_train, cls_valid_pop, y_valid, lam=1e-3,
        )
        torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)

        # Cleanup for next iter
        del cls_train_pop, cls_valid_pop, A_pop, B_pop, fitnesses

    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / 1e9
    avg_epoch_s = sum(epoch_times) / len(epoch_times)
    return peak_mem_gb, avg_epoch_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/cache_B"))
    p.add_argument("--dataset", type=str, default="esol")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pop-size", type=int, default=64)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=3, help="epochs per config")
    p.add_argument("--configs", type=str,
                   default="4,8;4,16;8,16;8,32;16,16;16,32;32,16;32,32;64,16;64,32",
                   help="semicolon-separated 'chunk,batch' pairs to test")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Benchmarking on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Load Unimol model + cache (do this ONCE) ----
    from unimol_tools.models import UniMolModel
    print(f"\nLoading UniMolModel ...")
    model = UniMolModel(output_dim=2, data_type="molecule", remove_hs=False)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    padding_idx = model.padding_idx

    targets = get_unimol_encoder_lora_targets(model)
    print(f"LoRA targets: {len(targets)} Linear weights")

    cache_dir = args.cache / args.dataset / f"seed_{args.split_seed}"
    train_data = torch.load(cache_dir / "train_conformers.pt", weights_only=False)
    valid_data = torch.load(cache_dir / "valid_conformers.pt", weights_only=False)
    train_conformers = train_data["conformers"]
    valid_conformers = valid_data["conformers"]
    y_train = train_data["targets"].to(device)
    y_valid = valid_data["targets"].to(device)
    print(f"Data: train={len(train_conformers)}, valid={len(valid_conformers)}")

    state = EGGROLLEncoderState(
        model=model, target_param_names=targets,
        rank=args.rank, sigma=args.sigma, device=device,
    ).to(device)
    vmap_fwd = make_vmap_forward(model, state, output_extractor=lambda d: d["cls_repr"])

    # ---- Parse configs ----
    configs = []
    for s in args.configs.split(";"):
        c, b = s.split(",")
        configs.append((int(c), int(b)))

    # ---- Run benchmarks ----
    print(f"\nConfig: pop_size={args.pop_size}, rank={args.rank}, n_epochs={args.n_epochs}")
    print(f"Testing {len(configs)} configurations:\n")
    print(f"{'chunk':>6}  {'batch':>6}  {'peak_mem_GB':>12}  {'epoch_s':>9}  {'speedup':>8}  status")
    print("-" * 60)

    results = []
    baseline_t = None
    for chunk_size, batch_size in configs:
        # Skip impossible configs
        if chunk_size > args.pop_size:
            print(f"{chunk_size:>6}  {batch_size:>6}  {'--':>12}  {'--':>9}  {'--':>8}  skip (chunk > pop)")
            continue
        try:
            peak_gb, epoch_s = benchmark_one_config(
                model, state, vmap_fwd, padding_idx,
                train_conformers, valid_conformers, y_train, y_valid,
                chunk_size=chunk_size, encoder_batch_size=batch_size,
                pop_size=args.pop_size, n_epochs=args.n_epochs, device=device,
            )
            if baseline_t is None:
                baseline_t = epoch_s
            speedup = baseline_t / epoch_s
            results.append({"chunk": chunk_size, "batch": batch_size,
                           "peak_mem_gb": peak_gb, "epoch_s": epoch_s, "speedup": speedup})
            print(f"{chunk_size:>6}  {batch_size:>6}  {peak_gb:>12.2f}  {epoch_s:>9.1f}  {speedup:>7.2f}x  OK")
        except torch.cuda.OutOfMemoryError:
            print(f"{chunk_size:>6}  {batch_size:>6}  {'OOM':>12}  {'--':>9}  {'--':>8}  OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{chunk_size:>6}  {batch_size:>6}  {'ERR':>12}  {'--':>9}  {'--':>8}  {type(e).__name__}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    # ---- Pick best & print recommendation ----
    if results:
        print("\n" + "=" * 60)
        best = min(results, key=lambda r: r["epoch_s"])
        print(f"BEST CONFIG: chunk={best['chunk']}, batch={best['batch']}")
        print(f"  Peak memory: {best['peak_mem_gb']:.2f} GB")
        print(f"  Epoch time: {best['epoch_s']:.1f}s ({best['speedup']:.2f}x speedup vs slowest)")
        print(f"\nTo use this config in training:")
        print(f"  --chunk-size {best['chunk']} --encoder-batch-size {best['batch']}")


if __name__ == "__main__":
    main()
# Experiment B — EGGROLL fine-tune Unimol-v1 encoder (reuses Step 1 pipeline)

## Nguyên tắc thiết kế

**Reuse Step 1's components để fair-compare:**
- ✅ Pre-split CSVs (`data/processed/{dataset}/seed_{N}/{dataset}_{train,valid,test}.csv`)
  → Same 81/9/10 split, same molecule allocation
- ✅ `ConformerGen` từ unimol_tools → same conformer generation
- ✅ `UniMolModel` từ unimol_tools → same encoder architecture, same pre-trained weights

**Chỉ khác Step 1 ở 2 điểm:**
- Optimizer: gradient descent → EGGROLL
- Head: MLP (Step 1) → ridge regression closed-form (Step 2)

## Step 1 baseline (cần beat)

| split_seed | Train | Valid | Test |
|---|---|---|---|
| 0 | 0.3304 | 0.7241 | 0.8579 |
| 1 | 0.3541 | 0.7569 | 0.8368 |
| 2 | 0.2460 | 0.6591 | 0.7579 |
| 3 | 0.3315 | 0.6718 | 0.8630 |
| 4 | 0.2459 | 0.6311 | 0.8376 |
| **Mean** | **0.30 ± 0.05** | **0.69 ± 0.05** | **0.83 ± 0.04** |

## Prerequisites: Step 1 preprocessing đã chạy

Trước khi chạy Experiment B, đảm bảo Step 1's preprocess_data.py đã chạy và tạo:
```
data/processed/esol/seed_{0..4}/esol_{train,valid,test}.csv
```

Nếu chưa có:
```powershell
foreach ($s in 0,1,2,3,4) {
    python scripts\preprocess_data.py --dataset esol --split-seed $s
}
```

## Workflow Experiment B

### Phase B.1: Verify trên seed 0 (~1.5-2.5 giờ)

```powershell
# 1. Generate conformers cho seed 0 (load từ Step 1's pre-split CSVs)
python experiments\exp_B_encoder\00_prepare_conformers.py `
    --dataset esol --split-seed 0

# 2. Training EGGROLL với tuned hyperparams
python experiments\exp_B_encoder\01_eggroll_encoder.py `
    --dataset esol --split-seed 0 `
    --pop-size 64 --chunk-size 8 `
    --rank 4 --sigma 0.01 --sigma-min 0.001 --sigma-decay cosine `
    --encoder-batch-size 16 `
    --epochs 300 --lr 1e-3 `
    --fitness-shaping rank `
    --eval-every 5 --patience 30
```

**Pass criterion**: test_rmse < 0.95 (Step 1 seed 0 đạt 0.86).

### Phase B.2: Full 5-seed run (sau khi Phase B.1 pass, ~7-13 giờ overnight)

```powershell
# Generate conformers cho 5 seeds
foreach ($s in 0,1,2,3,4) {
    python experiments\exp_B_encoder\00_prepare_conformers.py `
        --dataset esol --split-seed $s
}

# Train 5 seeds
foreach ($s in 0,1,2,3,4) {
    python experiments\exp_B_encoder\01_eggroll_encoder.py `
        --dataset esol --split-seed $s `
        --pop-size 64 --chunk-size 8 `
        --rank 4 --sigma 0.01 --sigma-min 0.001 --sigma-decay cosine `
        --encoder-batch-size 16 `
        --epochs 300 --lr 1e-3 `
        --fitness-shaping rank `
        --eval-every 5 --patience 30
}

# Aggregate & compare với Step 1
python experiments\exp_B_encoder\02_aggregate_compare.py `
    --dataset esol --results experiments\exp_B_encoder\results
```

## Tuning nếu Phase B.1 không đạt

Hyperparams ưu tiên thử (theo thứ tự impact):

1. **`--sigma 0.005 --sigma-min 0.0005`** — sigma nhỏ hơn nếu fitness curve dao động nhiều
2. **`--lr 5e-4`** — lr thấp hơn cho stable convergence
3. **`--pop-size 128 --chunk-size 8`** — pop lớn hơn giảm variance gradient
4. **`--ridge-lambda 1e-2`** — λ lớn nếu ridge ill-conditioned
5. **`--rank 8`** — rank lớn cho expressiveness
6. **`--layer-filter "10,11,12,13,14"`** — chỉ fine-tune 5 layer cuối (nhanh 3×)

## Output structure

```
data/cache_B/esol/seed_{N}/
├── train_conformers.pt
├── valid_conformers.pt
├── test_conformers.pt
└── meta.json

experiments/exp_B_encoder/results/esol/seed_{N}/
├── eggroll_encoder.pt
└── eggroll_encoder_summary.json

experiments/exp_B_encoder/results/esol/
└── aggregated_comparison.json
```

## Memory & wall-clock

Default config trên RTX 5070 Ti 16GB:
- Memory peak: ~5-6 GB
- Wall-clock: ~30s/epoch → 300 epochs ≈ 2.5h/seed → 5 seeds ≈ 12-13h
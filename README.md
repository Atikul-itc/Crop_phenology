# Phase 3 Baselines

Three baseline models on the phenology tensor bundles:

1. **Double-logistic curve fitting** on EVI (classical phenology). S2-only;
   predicts phases 15, 18, 21, 24 from SOS/PEAK/EOS heuristics.
2. **Random Forest** on engineered features. Runs all 5 stream configurations
   (S1 / S2 / Met / S1+S2 / All) and both label policies (all-10 / rank-1).
3. **Multi-stream LSTM** without MIL. Same 5×2 grid as RF, trained per-field
   with per-day soft Gaussian labels.

## Inputs

Pre-built tensor bundles from `phenology_pipeline.build_dataset`:

```
/fs1/home/atikul/Phenology/cache/tensors/
    phenology_2017.npz
    phenology_2018.npz
    phenology_2019.npz
    phenology_2020.npz
    phenology_2021.npz
```

## Quick Start

```bash
cd /fs1/home/atikul/Phenology

# 1. Double-logistic (fast, no GPU)
python -m phase3_baselines.baseline_doublelogistic \
    --tensors-dir cache/tensors \
    --output-dir phase3_outputs/dlog

# 2. Random Forest (CPU, ~30 min depending on n_estimators)
python -m phase3_baselines.baseline_rf \
    --tensors-dir cache/tensors \
    --output-dir phase3_outputs/rf \
    --n-estimators 200

# 3. LSTM (GPU strongly recommended)
python -m phase3_baselines.baseline_lstm \
    --tensors-dir cache/tensors \
    --output-dir phase3_outputs/lstm \
    --epochs 12 --hidden 64 --batch-size 64
```

To run only a subset of configurations (useful for first run / debugging):

```bash
python -m phase3_baselines.baseline_rf \
    --tensors-dir cache/tensors \
    --output-dir phase3_outputs/rf_smoke \
    --years 2019 2020 \
    --streams S1_only S2_only All \
    --policies rank1 \
    --n-estimators 50
```

## Outputs

Each baseline writes to its output dir:
- `*_results.csv`     per-fold, per-phase metrics (MAE, RMSE, R², within-5d, bias)
- `*_headline.csv`    macro-averaged summary per (stream, policy)
- `*_predictions.npz` per-record predictions for downstream analysis

## Evaluation Metrics

Per phase (and broken down by year/rank):
- `mae`       : mean absolute error in days
- `rmse`      : root MSE in days
- `r2`        : coefficient of determination
- `within_5d` : fraction of predictions within ±5 days
- `bias`      : mean signed error (pred − truth)
- `n`         : sample size

## Notes

- All baselines respect the same train/test split semantics: leave-one-year-out
  cross-validation. Records where the reference date is missing are excluded
  from that phase's metrics.
- "Policy" controls which fields are used for training/testing:
    - `all10`: every (XID, year) record (10 fields per station).
    - `rank1`: only the closest field per station.
  Comparing these two tells us how much label noise from the
  station-to-field ambiguity is hurting per-field training.
- The LSTM uses mask-aware GRU encoders per stream and ablates streams by
  zeroing inputs and masks for inactive streams — same architecture for all
  configurations, only the active set changes.

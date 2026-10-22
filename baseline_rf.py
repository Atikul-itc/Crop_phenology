"""
Random Forest baseline: predict day-of-window for each phase from engineered features.

Runs the 5-way stream ablation (S1 / S2 / Met / S1+S2 / All) and leave-one-year-out CV.
Also runs the "all-10" vs "rank-1 only" label-handling comparison.

Output: a pandas DataFrame of MAE/R²/within-5d per (config, test_year, phase),
plus saved per-record predictions for downstream analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from .dataset import (load_all_bundles, loyo_split, SelectionPolicy,
                       label_peak_days)
from .engineered_features import compute_for_bundle
from .evaluate import evaluate_predictions, print_evaluation


# Default stream configurations
S2_FEATS  = ["blue", "green", "red", "nir", "swir1", "swir2",
             "re1", "re2", "re3", "EVI", "MTCI", "NDRE", "CIrededge"]
S1_FEATS  = ["VV", "VH", "CR"]
CLIM_FEATS = ["mean_temp_celsius", "precipitation_mm", "cumulative_gdd"]

STREAM_CONFIGS = {
    "S1_only":   S1_FEATS,
    "S2_only":   S2_FEATS,
    "Met_only":  CLIM_FEATS,
    "S1+S2":     S1_FEATS + S2_FEATS,
    "All":       S1_FEATS + S2_FEATS + CLIM_FEATS,
}


def train_tabular(X_train: np.ndarray, Y_train: np.ndarray,
                   model_type: str = "hgb",
                   n_estimators: int = 200,
                   max_features: str = "sqrt",
                   n_jobs: int = -1,
                   random_state: int = 0):
    """Train one regressor per phase. model_type: 'rf' or 'hgb' (recommended)."""
    Y = Y_train.astype(np.float32).copy()
    Y[Y < 0] = np.nan

    P = Y.shape[1]
    models = []
    for j in range(P):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 50:
            models.append(None)
            continue
        if model_type == "rf":
            reg = RandomForestRegressor(n_estimators=n_estimators,
                                          max_features=max_features,
                                          n_jobs=n_jobs,
                                          random_state=random_state)
        elif model_type == "hgb":
            reg = HistGradientBoostingRegressor(
                max_iter=n_estimators, learning_rate=0.1,
                max_depth=None, max_leaf_nodes=63,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        reg.fit(X_train[m], Y[m, j])
        models.append(reg)
    return models


# Back-compat alias
train_rf = train_tabular


def predict_models(models: list, X: np.ndarray, n_phases: int) -> np.ndarray:
    """Run each phase-specific model. Returns (N, P) float predictions; NaN for unfit."""
    N = X.shape[0]
    out = np.full((N, n_phases), np.nan, dtype=np.float32)
    for j, m in enumerate(models):
        if m is None:
            continue
        out[:, j] = m.predict(X)
    return out


# Back-compat alias
predict_rf = predict_models


def run_one_config(bundles: dict, stream_name: str, feats_subset: list[str],
                   policy_name: str, policy: SelectionPolicy,
                   test_years: list[int],
                   n_estimators: int = 200) -> dict:
    """Run leave-one-year-out CV for a single (stream, policy) combo."""
    rows = []
    predictions = []

    phase_ids = next(iter(bundles.values()))["phase_ids"]

    for test_y in test_years:
        train_years = [y for y in bundles if y != test_y]
        if not train_years:
            print(f"  [skip] test={test_y}: no train years left")
            continue
        train_b, test_b = loyo_split(bundles, test_year=test_y, policy=policy)
        if len(train_b["xid"]) == 0 or len(test_b["xid"]) == 0:
            print(f"  [skip] {stream_name}/{policy_name}/test={test_y}: empty fold")
            continue

        # Engineered features
        X_tr, _    = compute_for_bundle(train_b, restrict_features=feats_subset)
        X_te, names = compute_for_bundle(test_b, restrict_features=feats_subset)

        # Targets: peak day per phase
        Y_tr = label_peak_days(train_b["Y"], train_b["label_mask"])  # (N_tr, P)
        Y_te = label_peak_days(test_b["Y"],  test_b["label_mask"])   # (N_te, P)

        print(f"  fitting {stream_name}/{policy_name}/test={test_y}: "
              f"X_tr={X_tr.shape}, X_te={X_te.shape}")
        models = train_rf(X_tr, Y_tr, n_estimators=n_estimators)
        pred_te = predict_rf(models, X_te, n_phases=Y_te.shape[1])

        # Evaluate
        report = evaluate_predictions(
            pred_days=pred_te, true_days=Y_te.astype(np.float32),
            phase_ids=phase_ids,
            years=test_b["harvest_year"],
            ranks=test_b["rank"],
        )
        for phase, m in report["per_phase"].iterrows():
            rows.append({
                "stream":   stream_name,
                "policy":   policy_name,
                "test_year": test_y,
                "phase":    int(phase),
                **m.to_dict()
            })

        # Save per-record predictions
        predictions.append({
            "test_year": test_y, "stream": stream_name, "policy": policy_name,
            "xid": test_b["xid"], "bag_id": test_b["bag_id"],
            "rank": test_b["rank"], "harvest_year": test_b["harvest_year"],
            "y_true": Y_te, "y_pred": pred_te,
        })

    return {"results": pd.DataFrame(rows), "predictions": predictions}


def main(tensors_dir: Path, output_dir: Path,
         years: list[int] = [2017, 2018, 2019, 2020, 2021],
         n_estimators: int = 200,
         streams: list[str] | None = None,
         policies: list[str] | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_all_bundles(tensors_dir, years=years)
    print(f"Loaded bundles for years: {sorted(bundles.keys())}")

    policy_map = {
        "all10":     SelectionPolicy(use_only_rank1=False),
        "rank1":     SelectionPolicy(use_only_rank1=True),
    }
    streams_to_run = streams or list(STREAM_CONFIGS.keys())
    policies_to_run = policies or list(policy_map.keys())

    all_results = []
    all_preds = []
    for s in streams_to_run:
        for pol_name in policies_to_run:
            print(f"\n--- Stream: {s} | Policy: {pol_name} ---")
            r = run_one_config(bundles, s, STREAM_CONFIGS[s],
                                pol_name, policy_map[pol_name],
                                test_years=years, n_estimators=n_estimators)
            all_results.append(r["results"])
            all_preds.extend(r["predictions"])

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(output_dir / "rf_results.csv", index=False)
    print(f"\nSaved: {output_dir / 'rf_results.csv'}")

    # Aggregate macro-MAE per (stream, policy)
    agg = (results_df.groupby(["stream", "policy", "phase"])
                     .agg(mae=("mae", "mean"),
                          r2 =("r2",  "mean"),
                          within_5d=("within_5d", "mean"),
                          n=("n", "sum"))
                     .reset_index())
    agg.to_csv(output_dir / "rf_results_agg.csv", index=False)

    # Headline summary table
    headline = (results_df.groupby(["stream", "policy"])
                          .agg(macro_mae=("mae", "mean"),
                               macro_r2 =("r2",  "mean"),
                               macro_within_5d=("within_5d", "mean"))
                          .reset_index())
    headline.to_csv(output_dir / "rf_headline.csv", index=False)
    print("\n=== Headline (macro over phases, mean across LOYO folds) ===")
    print(headline.round(3).to_string(index=False))

    # Save predictions as compressed npz
    np.savez_compressed(output_dir / "rf_predictions.npz",
                        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_xid":   p["xid"]    for p in all_preds},
                        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_bag":   p["bag_id"] for p in all_preds},
                        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_rank":  p["rank"]   for p in all_preds},
                        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_year":  p["harvest_year"] for p in all_preds},
                        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_yt":    p["y_true"] for p in all_preds},
                        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_yp":    p["y_pred"] for p in all_preds})
    return results_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors-dir", default="/fs1/home/atikul/Phenology/cache/tensors")
    ap.add_argument("--output-dir",  default="/fs1/home/atikul/Phenology/phase3_outputs/rf")
    ap.add_argument("--years",        nargs="+", type=int,
                    default=[2017, 2018, 2019, 2020, 2021])
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--streams",      nargs="+", default=None,
                    help=f"subset of: {list(STREAM_CONFIGS.keys())}")
    ap.add_argument("--policies",     nargs="+", default=None,
                    help="subset of: all10 rank1")
    args = ap.parse_args()
    main(Path(args.tensors_dir), Path(args.output_dir),
         years=args.years, n_estimators=args.n_estimators,
         streams=args.streams, policies=args.policies)

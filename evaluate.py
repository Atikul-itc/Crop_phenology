"""
Evaluation utilities for phenology baselines.

Predictions are days-of-window (int 0..T-1 or float). Truth is similar (from label_mask).

Metrics computed per phase, per (phase, year), and per (phase, rank):
  - MAE        : mean absolute error in days
  - RMSE       : root mean squared error in days
  - R²         : 1 - SS_res / SS_tot
  - within±5d  : fraction of predictions within 5 days of truth (matches our Gaussian σ)
  - bias       : mean signed error (pred − truth), to spot systematic shifts
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _safe(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_pred.mean()) ** 2)  # constant-predictor benchmark
    # Use variance of y_true for SS_tot (standard form)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 within_days: int = 5) -> dict:
    """Return MAE, RMSE, R2, within±d rate, bias, n."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0) & (y_pred >= 0)
    yt = y_true[mask].astype(np.float64)
    yp = y_pred[mask].astype(np.float64)
    if len(yt) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), f"within_{within_days}d": float("nan"),
                "bias": float("nan")}
    diff = yp - yt
    return {
        "n":     int(len(yt)),
        "mae":   float(np.mean(np.abs(diff))),
        "rmse":  float(np.sqrt(np.mean(diff ** 2))),
        "r2":    _r2(yt, yp),
        f"within_{within_days}d": float(np.mean(np.abs(diff) <= within_days)),
        "bias":  float(np.mean(diff)),
    }


def evaluate_predictions(
    pred_days: np.ndarray,        # (N, P) predicted day-of-window per phase, -1 means missing
    true_days: np.ndarray,        # (N, P) truth (same convention)
    phase_ids: list,              # length P
    years: np.ndarray | None = None,    # (N,) for per-year breakdown
    ranks: np.ndarray | None = None,    # (N,) for per-rank breakdown
    within_days: int = 5,
) -> dict:
    """Compute headline metrics + per-phase, per-(phase, year), per-(phase, rank) tables."""
    P = pred_days.shape[1]
    assert true_days.shape == pred_days.shape
    assert len(phase_ids) == P

    # Per-phase metrics
    per_phase = []
    for j, p in enumerate(phase_ids):
        m = core_metrics(true_days[:, j], pred_days[:, j], within_days)
        m["phase"] = int(p)
        per_phase.append(m)
    per_phase_df = pd.DataFrame(per_phase).set_index("phase")

    out = {"per_phase": per_phase_df}

    # Per-(phase, year)
    if years is not None:
        rows = []
        for j, p in enumerate(phase_ids):
            for yr in np.unique(years):
                idx = (years == yr)
                m = core_metrics(true_days[idx, j], pred_days[idx, j], within_days)
                m["phase"] = int(p)
                m["year"] = int(yr)
                rows.append(m)
        out["per_phase_year"] = pd.DataFrame(rows).set_index(["phase", "year"])

    # Per-(phase, rank)
    if ranks is not None:
        rows = []
        for j, p in enumerate(phase_ids):
            for r in np.unique(ranks):
                idx = (ranks == r)
                m = core_metrics(true_days[idx, j], pred_days[idx, j], within_days)
                m["phase"] = int(p)
                m["rank"]  = int(r)
                rows.append(m)
        out["per_phase_rank"] = pd.DataFrame(rows).set_index(["phase", "rank"])

    # Headline aggregate (macro-averaged over phases)
    valid_phase_mae = per_phase_df["mae"].dropna()
    out["headline"] = {
        "macro_mae":  float(valid_phase_mae.mean()),
        "macro_r2":   float(per_phase_df["r2"].dropna().mean()),
        f"macro_within_{within_days}d": float(per_phase_df[f"within_{within_days}d"].dropna().mean()),
        "n_phases_evaluated": int(valid_phase_mae.shape[0]),
    }
    return out


def print_evaluation(report: dict, title: str = "") -> None:
    print(f"\n{'='*70}")
    if title:
        print(title)
    print(f"{'='*70}")
    print("\n[Headline — macro over phases]")
    for k, v in report["headline"].items():
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.3f}")
        else:
            print(f"  {k:24s}: {v}")
    print("\n[Per phase]")
    print(report["per_phase"].round(3).to_string())
    if "per_phase_year" in report:
        print("\n[Per (phase, year)]")
        print(report["per_phase_year"]["mae"].unstack("year").round(2).to_string())
    if "per_phase_rank" in report:
        print("\n[Per (phase, rank) — MAE]")
        print(report["per_phase_rank"]["mae"].unstack("rank").round(2).to_string())

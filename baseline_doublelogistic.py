"""
Double-logistic curve fitting on EVI — classical phenology baseline.

For each (field, harvest_year), we:
  1. Extract the EVI time series, keeping only valid (observed) days.
  2. Restrict to a sensible spring–summer window (day 180–365 of window, ~Mar–Aug)
     to focus on the growing season for the harvest year.
  3. Fit a double-logistic curve:
        EVI(t) = m1 + m2 * (1/(1+exp(-m3*(t-m4))) - 1/(1+exp(-m5*(t-m6))))
     where m1 = winter baseline, m1+m2 = summer max, m3/m4 control green-up,
     m5/m6 control senescence.
  4. Extract characteristic dates:
        SOS  (start of season) = m4 (greenup inflection)
        EOS  (end of season)   = m6 (senescence inflection)
        PEAK = argmax of fitted curve

Mapping characteristic dates to BBCH phases (heuristic; reported as such):
  phase 15 (stem elong) ≈ SOS + a few days
  phase 18 (heading)    ≈ midpoint of (SOS, PEAK)
  phase 24 (harvest)    ≈ EOS

This baseline is honest about its limits: it estimates only 3 of the 7 phases
directly, and uses a fixed offset rule for those.

Output: per-record predicted days for phases 15, 18, 24. Other phases get NaN.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .dataset import load_all_bundles, label_peak_days
from .evaluate import evaluate_predictions


# Window for fitting: focus on growing season of the harvest year
# T=426 starts Sep 1 of (Y-1). Day 180 ≈ Feb 28 of Y; day 365 ≈ Aug 31 of Y.
FIT_LO, FIT_HI = 180, 365


def double_logistic(t, m1, m2, m3, m4, m5, m6):
    """Double-logistic seasonal model (Beck et al., 2006-style)."""
    s1 = 1.0 / (1.0 + np.exp(-m3 * (t - m4)))
    s2 = 1.0 / (1.0 + np.exp(-m5 * (t - m6)))
    return m1 + m2 * (s1 - s2)


def fit_one(t: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit double-logistic to a single field's EVI time series.

    Returns a dict of fitted params + derived dates, or None on failure.
    """
    if len(t) < 8:
        return None
    # Robust initial guesses
    y_low = np.percentile(y, 10)
    y_high = np.percentile(y, 90)
    t_mid = (t.min() + t.max()) / 2
    p0 = [y_low,                # m1
          max(y_high - y_low, 0.05),  # m2
          0.15,                  # m3 (greenup steepness)
          t_mid - 30,            # m4 (greenup midpoint)
          0.15,                  # m5
          t_mid + 30]            # m6 (senescence midpoint)
    bounds = ([-1, 0, 0.001, t.min()-30, 0.001, t.min()-30],
              [ 2, 2, 2.0,    t.max()+30, 2.0,    t.max()+30])

    try:
        res = least_squares(
            lambda p: double_logistic(t, *p) - y,
            x0=p0, bounds=bounds, method="trf", max_nfev=500
        )
    except Exception:
        return None
    if not res.success:
        return None

    m1, m2, m3, m4, m5, m6 = res.x
    # Sanity: m4 (green-up) should precede m6 (senescence)
    if m4 >= m6:
        return None
    # Peak: argmax of the fitted curve on a dense grid within the fitted window
    tt = np.linspace(t.min(), t.max(), 400)
    yy = double_logistic(tt, *res.x)
    peak = float(tt[np.argmax(yy)])
    return {
        "m": res.x.tolist(),
        "SOS": float(m4),
        "EOS": float(m6),
        "PEAK": peak,
        "residual_rms": float(np.sqrt(np.mean(res.fun ** 2))),
    }


def predict_for_bundle(bundle: dict) -> np.ndarray:
    """Predict day-of-window for each phase using curve-fit dates.

    Returns (N, P) float predictions; NaN for unestimated phases.
    """
    feats = bundle["feature_names"]
    if "EVI" not in feats:
        raise KeyError("Bundle lacks EVI feature")
    evi_idx = feats.index("EVI")

    X = bundle["X"][:, :, evi_idx]   # (N, T)
    M = bundle["mask"][:, :, evi_idx]
    phase_ids = bundle["phase_ids"]
    P = len(phase_ids)
    N, T = X.shape

    pred = np.full((N, P), np.nan, dtype=np.float32)
    n_fit_ok = 0
    n_attempt = 0

    # Phase index map
    phase_idx = {p: i for i, p in enumerate(phase_ids)}

    # Limit to the fit window
    t_all = np.arange(T, dtype=np.float32)
    for i in range(N):
        m_i = M[i, FIT_LO:FIT_HI]
        if m_i.sum() < 8:
            continue
        t = t_all[FIT_LO:FIT_HI][m_i]
        y = X[i, FIT_LO:FIT_HI][m_i]
        n_attempt += 1
        fit = fit_one(t, y)
        if fit is None:
            continue
        n_fit_ok += 1
        sos, eos, peak = fit["SOS"], fit["EOS"], fit["PEAK"]
        # Heuristic mapping:
        if 15 in phase_idx:
            pred[i, phase_idx[15]] = sos             # SOS ≈ stem elongation start
        if 18 in phase_idx:
            pred[i, phase_idx[18]] = 0.5 * (sos + peak)  # midpoint
        if 21 in phase_idx:
            pred[i, phase_idx[21]] = 0.5 * (peak + eos)
        if 24 in phase_idx:
            pred[i, phase_idx[24]] = eos             # EOS ≈ harvest

    print(f"  curve-fit: {n_fit_ok}/{n_attempt} succeeded "
          f"({100*n_fit_ok/max(n_attempt,1):.1f}%)")
    return pred


def main(tensors_dir: Path, output_dir: Path,
         years: list[int] = [2017, 2018, 2019, 2020, 2021]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_all_bundles(tensors_dir, years=years)
    print(f"Loaded bundles for years: {sorted(bundles.keys())}")

    rows = []
    for y in years:
        if y not in bundles:
            continue
        b = bundles[y]
        print(f"\nFitting double-logistic for {y} (N={len(b['xid'])}) ...")
        pred = predict_for_bundle(b)
        truth = label_peak_days(b["Y"], b["label_mask"]).astype(np.float32)
        # Mark NaN predictions as -1 for evaluate compatibility
        pred_eval = np.where(np.isnan(pred), -1, pred).astype(np.float32)

        report = evaluate_predictions(
            pred_days=pred_eval, true_days=truth,
            phase_ids=b["phase_ids"],
            years=b["harvest_year"], ranks=b["rank"],
        )
        for phase, m in report["per_phase"].iterrows():
            rows.append({"test_year": int(y), "phase": int(phase),
                         **m.to_dict()})

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "doublelogistic_results.csv", index=False)

    # Headline (only for phases the method can predict)
    print("\n=== Double-logistic — per phase, per year (MAE) ===")
    pivot = results.pivot(index="phase", columns="test_year", values="mae").round(2)
    print(pivot.to_string())

    print("\n=== Macro MAE per year ===")
    print(results.groupby("test_year")["mae"].mean().round(2).to_string())

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors-dir", default="/fs1/home/atikul/Phenology/cache/tensors")
    ap.add_argument("--output-dir",  default="/fs1/home/atikul/Phenology/phase3_outputs/dlog")
    ap.add_argument("--years",       nargs="+", type=int,
                    default=[2017, 2018, 2019, 2020, 2021])
    args = ap.parse_args()
    main(Path(args.tensors_dir), Path(args.output_dir), years=args.years)

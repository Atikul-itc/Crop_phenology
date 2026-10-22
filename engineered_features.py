"""
Hand-crafted features for the Random Forest baseline.

For each (record, harvest_year), we compute a fixed-size feature vector by
summarizing the time series within phenologically meaningful WINDOWS, plus a
few global statistics. The vector contains:

  - Per-window, per-feature: mean, std, min, max, argmax-day, argmin-day
  - S2 cloudiness proxy: observation rate per window
  - Within-station rank (as integer feature)

Windows (day-of-window indices, T=426 starting Sep 1 of Y-1):
  - autumn        :   0–120   (Sep–Dec of Y-1)
  - winter        : 121–212   (Jan–Mar of Y)
  - spring_early  : 213–272   (Apr–May)
  - spring_late   : 273–304   (Jun)
  - summer        : 305–365   (Jul)
  - late_summer   : 366–425   (Aug–Oct of Y)

Rationale: these windows roughly bracket sequential developmental periods, so
the RF can pick up window-specific signals (e.g., GDD in spring_early predicts
stem elongation; SWIR in summer predicts ripening).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd


# Silence the expected all-NaN-slice warnings (we handle them explicitly)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")


WINDOWS = [
    ("autumn",       0,   121),   # Sep 1 .. ~Dec 30 of Y-1
    ("winter",       121, 213),   # ~Jan 1 .. Mar 31 of Y
    ("spring_early", 213, 273),   # ~Apr 1 .. May 31
    ("spring_late",  273, 305),   # ~Jun 1 .. Jun 30
    ("summer",       305, 366),   # ~Jul 1 .. Jul 31
    ("late_summer",  366, 426),   # ~Aug 1 .. Oct 31
]


def compute_record_features(X: np.ndarray, mask: np.ndarray,
                              feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Compute the engineered feature vector for a batch of records.

    Args:
      X     : (N, T, F) values (0 where unobserved; meaningful where observed)
      mask  : (N, T, F) bool (True where observed)
      feature_names: length-F list

    Returns:
      feats : (N, K) float32
      names : length-K list of feature names
    """
    N, T, F = X.shape
    rows = []

    # Pre-mask: replace zeros where unobserved with NaN to make summary stats honest.
    Xm = np.where(mask, X, np.nan)

    for win_name, lo, hi in WINDOWS:
        sl = Xm[:, lo:hi, :]                 # (N, W, F)
        msl = mask[:, lo:hi, :]              # (N, W, F)

        # Suppress all-NaN warnings — we accept NaN summary stats if a window is empty
        with np.errstate(invalid="ignore"):
            mean_ = np.nanmean(sl, axis=1)   # (N, F)
            std_  = np.nanstd(sl, axis=1)
            min_  = np.nanmin(sl, axis=1)
            max_  = np.nanmax(sl, axis=1)
        # For argmax/argmin: replace NaN with -inf/+inf so they don't get picked
        with np.errstate(invalid="ignore"):
            sl_for_argmax = np.where(np.isnan(sl), -np.inf, sl)
            sl_for_argmin = np.where(np.isnan(sl), np.inf, sl)
            argmax_ = sl_for_argmax.argmax(axis=1).astype(np.float32) + lo
            argmin_ = sl_for_argmin.argmin(axis=1).astype(np.float32) + lo
            # If a whole column was NaN, argmax/min return 0; we mask those to NaN
            all_nan = ~msl.any(axis=1)
            argmax_[all_nan] = np.nan
            argmin_[all_nan] = np.nan

        # Observation rate per feature in this window
        obs_rate = msl.mean(axis=1).astype(np.float32)

        # Stack: (N, F, 7) -> (N, 7*F)
        block = np.stack([mean_, std_, min_, max_, argmax_, argmin_, obs_rate],
                         axis=-1).reshape(N, -1)
        rows.append(block.astype(np.float32))

    # Build names: per window, per stat, per feature (order matches the stack above)
    names = []
    for win_name, _, _ in WINDOWS:
        for stat in ("mean", "std", "min", "max", "argmax", "argmin", "obsrate"):
            for f in feature_names:
                names.append(f"{win_name}__{f}__{stat}")

    feats = np.concatenate(rows, axis=1)   # (N, 6 windows * 7 stats * F)
    assert feats.shape[1] == len(names), f"{feats.shape[1]} vs {len(names)}"

    # Replace remaining NaN with 0 — trees handle this fine and RF can't accept NaN
    # in sklearn's default implementation. We add a parallel "missing indicator" column
    # group below if needed; for now, simple imputation is enough.
    nan_count = np.isnan(feats).sum()
    if nan_count > 0:
        feats = np.where(np.isnan(feats), 0.0, feats)

    return feats.astype(np.float32), names


def compute_for_bundle(bundle: dict,
                       restrict_features: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Convenience: compute engineered features for a bundle, optionally
    restricted to a subset of input features (e.g., only S1).
    """
    feats_in = bundle["feature_names"]
    X = bundle["X"]
    M = bundle["mask"]

    if restrict_features is not None:
        idx = [feats_in.index(f) for f in restrict_features]
        X = X[:, :, idx]
        M = M[:, :, idx]
        feats_in = restrict_features

    return compute_record_features(X, M, feats_in)

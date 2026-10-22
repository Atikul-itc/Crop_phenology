"""
PyTorch-compatible dataset + split utilities for Phase 3 baselines.

A "record" is one (XID, harvest_year) entry from the saved .npz bundles.

Splits:
  - LOYO (leave-one-year-out): train on 4 years, test on 1.
  - "all-10 fields" vs "rank-1 only": two ways of using each year's data.

The dataset returns numpy arrays for sklearn-based baselines AND tensors for the LSTM
baseline. To keep the code simple, we lazy-import torch only when needed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np


# ---------- BUNDLE LOADING ----------
def load_year_bundle(npz_path: Path) -> dict:
    """Load a per-year .npz bundle produced by phenology_pipeline.build_dataset."""
    z = np.load(npz_path, allow_pickle=False)
    return {
        "X":            z["X"],                # (N, T, F) float32
        "mask":         z["mask"],             # (N, T, F) bool
        "Y":            z["Y"],                # (N, T, P) float32 — soft Gaussian labels
        "label_mask":   z["label_mask"],       # (N, P)    bool
        "bag_id":       z["bag_id"],           # (N,) int64 (Stations_i)
        "rank":         z["rank"],             # (N,) int32 (1..10)
        "harvest_year": z["harvest_year"],     # (N,) int32
        "xid":          z["xid"],              # (N,) int64
        "feature_names": [str(s) for s in z["feature_names"]],
        "phase_ids":    z["phase_ids"].astype(int).tolist(),
        "T":            int(z["T"]),
    }


def load_all_bundles(tensors_dir: Path,
                      years: Iterable[int] = range(2017, 2022)) -> dict[int, dict]:
    bundles = {}
    for y in years:
        p = Path(tensors_dir) / f"phenology_{y}.npz"
        if p.exists():
            bundles[y] = load_year_bundle(p)
        else:
            print(f"[warn] missing bundle for {y}: {p}")
    return bundles


# ---------- CONCATENATION ----------
def concat_bundles(bundles: dict[int, dict],
                    years: Iterable[int]) -> dict:
    """Concatenate bundle arrays across the requested years.

    Assumes T and feature ordering are identical across years (verified by pipeline).
    """
    years = list(years)
    parts = [bundles[y] for y in years if y in bundles]
    if not parts:
        raise ValueError(f"No bundles available for years {years}")

    T = parts[0]["T"]
    feats = parts[0]["feature_names"]
    phase_ids = parts[0]["phase_ids"]
    for b in parts[1:]:
        assert b["T"] == T, f"T mismatch: {b['T']} vs {T}"
        assert b["feature_names"] == feats, "feature names differ across years"
        assert b["phase_ids"] == phase_ids, "phase ids differ across years"

    def cat(key, dtype=None):
        arr = np.concatenate([b[key] for b in parts], axis=0)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    return {
        "X":            cat("X"),
        "mask":         cat("mask"),
        "Y":            cat("Y"),
        "label_mask":   cat("label_mask"),
        "bag_id":       cat("bag_id"),
        "rank":         cat("rank"),
        "harvest_year": cat("harvest_year"),
        "xid":          cat("xid"),
        "feature_names": feats,
        "phase_ids":    phase_ids,
        "T":            T,
    }


# ---------- RECORD SELECTION ----------
@dataclass
class SelectionPolicy:
    """How to choose records from each (station, year)."""
    use_only_rank1: bool = False
    require_all_phases: bool = False           # drop records missing any phase
    require_phases: tuple[int, ...] | None = None  # require specific phases present


def apply_selection(bundle: dict, policy: SelectionPolicy) -> dict:
    """Filter records according to the policy. Returns a new bundle dict."""
    keep = np.ones(len(bundle["xid"]), dtype=bool)
    if policy.use_only_rank1:
        keep &= (bundle["rank"] == 1)
    if policy.require_all_phases:
        keep &= bundle["label_mask"].all(axis=1)
    if policy.require_phases:
        idx_map = {p: i for i, p in enumerate(bundle["phase_ids"])}
        for p in policy.require_phases:
            if p not in idx_map:
                raise KeyError(f"Phase {p} not in bundle phase_ids {bundle['phase_ids']}")
            keep &= bundle["label_mask"][:, idx_map[p]]
    out = dict(bundle)
    for k in ("X", "mask", "Y", "label_mask", "bag_id", "rank", "harvest_year", "xid"):
        out[k] = bundle[k][keep]
    return out


# ---------- LEAVE-ONE-YEAR-OUT SPLITS ----------
def loyo_split(bundles: dict[int, dict],
                test_year: int,
                policy: SelectionPolicy | None = None) -> tuple[dict, dict]:
    """Return (train_bundle, test_bundle) for a leave-one-year-out split."""
    policy = policy or SelectionPolicy()
    train_years = [y for y in bundles if y != test_year]
    if not train_years:
        raise ValueError(f"No train years left after removing {test_year}")
    train = concat_bundles(bundles, train_years)
    test  = concat_bundles(bundles, [test_year])
    return apply_selection(train, policy), apply_selection(test, policy)


# ---------- LABEL EXTRACTION ----------
def label_peak_days(Y: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
    """Extract the argmax (day index) of each phase's Gaussian curve.

    Args:
      Y: (N, T, P)
      label_mask: (N, P) bool — True if phase has a valid reference date
    Returns:
      peaks: (N, P) int32 — -1 where label_mask is False
    """
    peaks = Y.argmax(axis=1).astype(np.int32)        # (N, P)
    peaks[~label_mask] = -1
    return peaks


# ---------- TORCH DATASET (lazy-import) ----------
def make_torch_dataset(bundle: dict, return_mask: bool = True):
    """Wrap a bundle in a torch.utils.data.Dataset."""
    import torch
    from torch.utils.data import Dataset

    X     = torch.from_numpy(bundle["X"])           # (N, T, F) float32
    M     = torch.from_numpy(bundle["mask"]).to(torch.float32)
    Y     = torch.from_numpy(bundle["Y"])           # (N, T, P) float32
    LM    = torch.from_numpy(bundle["label_mask"])  # (N, P) bool
    bag   = torch.from_numpy(bundle["bag_id"])
    rank_ = torch.from_numpy(bundle["rank"])
    yr    = torch.from_numpy(bundle["harvest_year"])
    xid   = torch.from_numpy(bundle["xid"])

    class _BundleDataset(Dataset):
        def __len__(self):
            return X.shape[0]
        def __getitem__(self, i):
            item = {
                "X":      X[i],
                "Y":      Y[i],
                "label_mask": LM[i],
                "bag_id": bag[i],
                "rank":   rank_[i],
                "harvest_year": yr[i],
                "xid":    xid[i],
            }
            if return_mask:
                item["mask"] = M[i]
            return item

    return _BundleDataset()

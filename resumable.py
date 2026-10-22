"""
Lightweight helpers for resumable baselines in Jupyter.

Pattern: each fold (or each fit) writes a single row to a CSV as soon as it's done.
On restart, we read the CSV, figure out which (stream, policy, test_year) tuples
are already done, and skip them. So you can interrupt at any time and resume.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd


def append_row(csv_path: Path, row: dict) -> None:
    """Append a single row to a CSV. Creates header if file doesn't exist."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def already_done(csv_path: Path, key_cols: list[str]) -> set:
    """Return set of tuples (sorted by key_cols) already present in the CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if not all(c in df.columns for c in key_cols):
        return set()
    # Each tuple key
    return {tuple(r) for r in df[key_cols].drop_duplicates().itertuples(index=False)}


def save_pred_npz(npz_path: Path, key: str, **arrays) -> None:
    """Save per-fold predictions, accumulating into a single npz over time.

    Each call adds a set of arrays keyed by `{key}__{name}`.
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            existing = {k: z[k] for k in z.files}
    for name, arr in arrays.items():
        existing[f"{key}__{name}"] = arr
    np.savez_compressed(npz_path, **existing)

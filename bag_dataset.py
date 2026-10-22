"""
Bagged dataset for MIL training.

A "bag" is a station (Stations_i). Each bag contains up to 10 fields (XIDs).

The dataset returns bags as variable-size tensors with a per-bag mask. To make
batching efficient, we pad bags to a fixed size (B_MAX=10) and provide a bag-mask
indicating which slots are valid.

For each bag we have:
    X         : (B_MAX, T, F)   field-level features
    feat_mask : (B_MAX, T, F)   field-level observation mask
    bag_mask  : (B_MAX,)        True for valid fields, False for padding
    rank      : (B_MAX,)        within-station rank (1..10, 0 for padding)
    xid       : (B_MAX,)        unique field IDs (0 for padding)
    Y         : (T, P)          STATION-LEVEL Gaussian labels (shared by all fields)
    label_mask: (P,)            which phases have reference dates for this station
    bag_id    : scalar          Stations_i
    year      : scalar          harvest year
"""
from __future__ import annotations
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset


B_MAX = 10   # max fields per bag (matches DWD network design)


def build_bags(bundle: dict, b_max: int = B_MAX) -> dict:
    """Group a flat (per-field) bundle into bags (per-station).

    Returns a dict of arrays where the first dimension is N_bags (=N_stations).
    Fields with the same (bag_id, harvest_year) form one bag.
    """
    bag_keys = list(zip(bundle["bag_id"].tolist(), bundle["harvest_year"].tolist()))
    # Map each (station, year) pair to a list of original-index positions
    groups: dict = {}
    for i, k in enumerate(bag_keys):
        groups.setdefault(k, []).append(i)

    n_bags = len(groups)
    T      = bundle["X"].shape[1]
    F      = bundle["X"].shape[2]
    P      = bundle["Y"].shape[2]

    bag_X       = np.zeros((n_bags, b_max, T, F), dtype=np.float32)
    bag_feat_M  = np.zeros((n_bags, b_max, T, F), dtype=bool)
    bag_mask    = np.zeros((n_bags, b_max), dtype=bool)
    bag_rank    = np.zeros((n_bags, b_max), dtype=np.int32)
    bag_xid     = np.zeros((n_bags, b_max), dtype=np.int64)

    bag_Y       = np.zeros((n_bags, T, P), dtype=np.float32)
    bag_LM      = np.zeros((n_bags, P), dtype=bool)
    bag_year    = np.zeros(n_bags, dtype=np.int32)
    bag_sid     = np.zeros(n_bags, dtype=np.int64)

    for bi, (key, idxs) in enumerate(groups.items()):
        sid, year = key
        bag_sid[bi]  = sid
        bag_year[bi] = year

        # Sort by rank for reproducibility (rank 1 always in slot 0 when present)
        idxs_sorted = sorted(idxs, key=lambda i: int(bundle["rank"][i]))
        n = min(len(idxs_sorted), b_max)
        for s in range(n):
            i = idxs_sorted[s]
            bag_X[bi, s]      = bundle["X"][i]
            bag_feat_M[bi, s] = bundle["mask"][i]
            bag_mask[bi, s]   = True
            bag_rank[bi, s]   = bundle["rank"][i]
            bag_xid[bi, s]    = bundle["xid"][i]

        # Station-level label: identical across fields (they share the same reference dates).
        # Use the first valid field's labels (they should all be the same).
        first_idx = idxs_sorted[0]
        bag_Y[bi]  = bundle["Y"][first_idx]
        bag_LM[bi] = bundle["label_mask"][first_idx]

    return {
        "X": bag_X, "feat_mask": bag_feat_M, "bag_mask": bag_mask,
        "rank": bag_rank, "xid": bag_xid,
        "Y": bag_Y, "label_mask": bag_LM,
        "bag_id": bag_sid, "harvest_year": bag_year,
        "feature_names": bundle["feature_names"],
        "phase_ids":     bundle["phase_ids"],
        "T":             bundle["T"],
        "B_MAX":         b_max,
    }


class BaggedDataset(Dataset):
    """Wraps a bagged bundle for PyTorch."""
    def __init__(self, bags: dict):
        self.X       = torch.from_numpy(bags["X"])
        self.feat_M  = torch.from_numpy(bags["feat_mask"])      # bool
        self.bagM    = torch.from_numpy(bags["bag_mask"])       # bool
        self.rank    = torch.from_numpy(bags["rank"])
        self.xid     = torch.from_numpy(bags["xid"])
        self.Y       = torch.from_numpy(bags["Y"])
        self.LM      = torch.from_numpy(bags["label_mask"])     # bool
        self.bag_id  = torch.from_numpy(bags["bag_id"])
        self.year    = torch.from_numpy(bags["harvest_year"])
        self.N       = self.X.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, i: int) -> dict:
        return {
            "X":         self.X[i],            # (B, T, F)
            "feat_mask": self.feat_M[i],       # (B, T, F) bool
            "bag_mask":  self.bagM[i],         # (B,) bool
            "rank":      self.rank[i],         # (B,)
            "xid":       self.xid[i],          # (B,)
            "Y":         self.Y[i],            # (T, P)
            "label_mask":self.LM[i],           # (P,) bool
            "bag_id":    self.bag_id[i],
            "year":      self.year[i],
        }


# ---------- HELPERS ----------
def iter_bag_batches(ds: BaggedDataset, batch_size: int, shuffle: bool,
                      device: torch.device, generator: torch.Generator | None = None):
    """Mini-batch iteration over bags. Returns tensors moved to device.

    Reusing the per-batch-streaming pattern from Phase 3 to keep GPU memory low.
    Mask stays bool until cast on-device.
    """
    idx = torch.randperm(ds.N, generator=generator) if shuffle else torch.arange(ds.N)
    for i in range(0, ds.N, batch_size):
        sl = idx[i:i+batch_size]
        x     = ds.X[sl].to(device, non_blocking=True)
        fmask = ds.feat_M[sl].to(device, non_blocking=True).to(torch.float32)
        bmask = ds.bagM[sl].to(device, non_blocking=True)        # bool
        y     = ds.Y[sl].to(device, non_blocking=True)
        lm    = ds.LM[sl].to(device, non_blocking=True)
        rank  = ds.rank[sl].to(device, non_blocking=True)
        xid   = ds.xid[sl].to(device, non_blocking=True)
        yield {"X": x, "feat_mask": fmask, "bag_mask": bmask,
               "Y": y, "label_mask": lm, "rank": rank, "xid": xid, "idx": sl}

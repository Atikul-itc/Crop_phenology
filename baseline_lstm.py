"""
Simple multi-stream LSTM baseline (no MIL).

Architecture:
  Each input stream (S1, S2, climate) has its own GRU/LSTM encoder.
  Hidden states are concatenated per timestep.
  A linear head produces P-channel per-day phase probabilities (via sigmoid).
  Training target: soft Gaussian per-day labels.

Per-field training: assume the station's reference date applies to every field.
Comparison policy "all10" vs "rank1" controls whether we train on all fields or
only the closest.

Loss: per-day per-phase BCE (binary cross-entropy with logits) where label_mask
is True (i.e., we only contribute loss for phases that have a reference date).
For "missing phase" we zero out the gradient.

Inference: extract argmax of the per-day predicted probability for each phase
to get the predicted day-of-window. Same metrics as RF.

Stream ablation: pass an `active_streams` set into the model. Inactive streams
have their input zeroed (and the corresponding mask is zeroed). This is a
clean way to do the 5-way ablation without changing the architecture.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (load_all_bundles, loyo_split, SelectionPolicy,
                       label_peak_days, make_torch_dataset)
from .evaluate import evaluate_predictions


# ---------- FEATURE GROUPING ----------
S2_FEATS = ["blue", "green", "red", "nir", "swir1", "swir2",
            "re1", "re2", "re3", "EVI", "MTCI", "NDRE", "CIrededge"]
S1_FEATS = ["VV", "VH", "CR"]
CLIM_FEATS = ["mean_temp_celsius", "precipitation_mm", "cumulative_gdd"]


def stream_indices(feature_names: list[str]) -> dict:
    return {
        "S2":  [feature_names.index(f) for f in S2_FEATS  if f in feature_names],
        "S1":  [feature_names.index(f) for f in S1_FEATS  if f in feature_names],
        "Met": [feature_names.index(f) for f in CLIM_FEATS if f in feature_names],
    }


# ---------- MODEL ----------
class StreamEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # Input is (raw values + mask) concatenated → 2*in_dim
        self.gru = nn.GRU(input_size=2 * in_dim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, x, m):
        # x: (B, T, F), m: (B, T, F)
        inp = torch.cat([x * m, m], dim=-1)   # mask-aware input
        h, _ = self.gru(inp)
        return h


class MultiStreamLSTM(nn.Module):
    def __init__(self, feature_names: list[str],
                 hidden: int = 64,
                 num_layers: int = 1,
                 n_phases: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_names = feature_names
        self.idx = stream_indices(feature_names)
        self.streams = nn.ModuleDict()
        total_dim = 0
        for name, idxs in self.idx.items():
            if len(idxs) == 0:
                continue
            enc = StreamEncoder(in_dim=len(idxs), hidden=hidden,
                                 num_layers=num_layers, dropout=dropout)
            self.streams[name] = enc
            total_dim += enc.out_dim
        self.head = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_phases),
        )

    def forward(self, x, m, active=("S1", "S2", "Met")):
        outs = []
        for name in self.streams:
            if name in active:
                idxs = self.idx[name]
                xs = x[:, :, idxs]
                ms = m[:, :, idxs]
            else:
                # Inactive stream: zero input + zero mask
                idxs = self.idx[name]
                xs = torch.zeros_like(x[:, :, idxs])
                ms = torch.zeros_like(m[:, :, idxs])
            outs.append(self.streams[name](xs, ms))
        h = torch.cat(outs, dim=-1)              # (B, T, sum_hidden)
        logits = self.head(h)                    # (B, T, P)
        return logits


# ---------- LOSS ----------
def soft_label_loss(logits: torch.Tensor, target: torch.Tensor,
                     label_mask: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits per (B, T, P), masked over phases without reference dates."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )                                       # (B, T, P)
    # Mask: (B, P) -> (B, 1, P) so we can broadcast over T
    mask = label_mask.unsqueeze(1).float()
    loss = (bce * mask).sum() / (mask.sum() * logits.shape[1] + 1e-8)
    return loss


# ---------- INFERENCE ----------
@torch.no_grad()
def predict_argmax(model: MultiStreamLSTM, loader: DataLoader,
                    active: tuple, device: torch.device,
                    return_arrays: bool = True) -> tuple:
    model.eval()
    all_pred = []
    all_truth = []
    all_xid = []
    all_bag = []
    all_rank = []
    all_year = []
    for batch in loader:
        x = batch["X"].to(device)
        m = batch["mask"].to(device)
        y = batch["Y"].to(device)
        lm = batch["label_mask"].to(device)
        logits = model(x, m, active=active)            # (B, T, P)
        # Predicted day = argmax along T
        pred_day = logits.argmax(dim=1).cpu().numpy()  # (B, P)
        truth_day = y.argmax(dim=1).cpu().numpy()      # (B, P)
        truth_day[~lm.cpu().numpy()] = -1
        all_pred.append(pred_day)
        all_truth.append(truth_day)
        all_xid.append(batch["xid"].cpu().numpy())
        all_bag.append(batch["bag_id"].cpu().numpy())
        all_rank.append(batch["rank"].cpu().numpy())
        all_year.append(batch["harvest_year"].cpu().numpy())
    return (np.concatenate(all_pred),  np.concatenate(all_truth),
            np.concatenate(all_xid),   np.concatenate(all_bag),
            np.concatenate(all_rank),  np.concatenate(all_year))


# ---------- TRAINING ----------
def train_one_fold(train_b: dict, test_b: dict,
                    active: tuple,
                    hidden: int = 64, num_layers: int = 1,
                    epochs: int = 12, batch_size: int = 64,
                    lr: float = 1e-3, weight_decay: float = 1e-5,
                    device: torch.device = None,
                    verbose: bool = True) -> tuple:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = make_torch_dataset(train_b)
    test_ds  = make_torch_dataset(test_b)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type=="cuda"))

    n_phases = train_b["Y"].shape[2]
    model = MultiStreamLSTM(feature_names=train_b["feature_names"],
                              hidden=hidden, num_layers=num_layers,
                              n_phases=n_phases).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        total = 0.0
        nb = 0
        for batch in train_loader:
            x = batch["X"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            y = batch["Y"].to(device, non_blocking=True)
            lm = batch["label_mask"].to(device, non_blocking=True)
            logits = model(x, m, active=active)
            loss = soft_label_loss(logits, y, lm)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        if verbose:
            print(f"    epoch {ep+1:2d}/{epochs}: train_loss={total/max(nb,1):.4f}")

    pred_te, truth_te, xid, bag, rank, yr = predict_argmax(
        model, test_loader, active, device
    )
    return pred_te, truth_te, xid, bag, rank, yr


# ---------- ORCHESTRATOR ----------
STREAM_CONFIGS = {
    "S1_only":  ("S1",),
    "S2_only":  ("S2",),
    "Met_only": ("Met",),
    "S1+S2":    ("S1", "S2"),
    "All":      ("S1", "S2", "Met"),
}


def main(tensors_dir: Path, output_dir: Path,
         years: list[int] = [2017, 2018, 2019, 2020, 2021],
         epochs: int = 12, hidden: int = 64, batch_size: int = 64,
         streams: list[str] | None = None,
         policies: list[str] | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_all_bundles(tensors_dir, years=years)
    print(f"Loaded bundles for years: {sorted(bundles.keys())}")
    phase_ids = next(iter(bundles.values()))["phase_ids"]

    policy_map = {
        "all10": SelectionPolicy(use_only_rank1=False),
        "rank1": SelectionPolicy(use_only_rank1=True),
    }
    streams_to_run = streams or list(STREAM_CONFIGS.keys())
    policies_to_run = policies or list(policy_map.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    pred_records = []

    for stream_name in streams_to_run:
        active = STREAM_CONFIGS[stream_name]
        for pol_name in policies_to_run:
            print(f"\n--- LSTM | Stream: {stream_name} | Policy: {pol_name} ---")
            for test_y in years:
                train_b, test_b = loyo_split(bundles, test_y, policy_map[pol_name])
                if len(train_b["xid"]) == 0 or len(test_b["xid"]) == 0:
                    continue
                print(f"  test_year={test_y}: train N={len(train_b['xid'])}, "
                      f"test N={len(test_b['xid'])}")
                pred, truth, xid, bag, rank, yr = train_one_fold(
                    train_b, test_b, active,
                    hidden=hidden, epochs=epochs, batch_size=batch_size,
                    device=device, verbose=True
                )
                report = evaluate_predictions(
                    pred_days=pred.astype(np.float32),
                    true_days=truth.astype(np.float32),
                    phase_ids=phase_ids,
                    years=yr, ranks=rank,
                )
                for phase, m in report["per_phase"].iterrows():
                    rows.append({
                        "stream": stream_name, "policy": pol_name,
                        "test_year": int(test_y), "phase": int(phase),
                        **m.to_dict()
                    })
                pred_records.append({
                    "stream": stream_name, "policy": pol_name,
                    "test_year": test_y, "xid": xid, "bag": bag, "rank": rank,
                    "year": yr, "yt": truth, "yp": pred,
                })

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "lstm_results.csv", index=False)

    headline = (results.groupby(["stream", "policy"])
                       .agg(macro_mae=("mae", "mean"),
                            macro_r2 =("r2",  "mean"),
                            macro_within_5d=("within_5d", "mean"))
                       .reset_index())
    headline.to_csv(output_dir / "lstm_headline.csv", index=False)
    print("\n=== LSTM Headline (macro over phases, mean across LOYO folds) ===")
    print(headline.round(3).to_string(index=False))

    # Save predictions
    np.savez_compressed(
        output_dir / "lstm_predictions.npz",
        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_xid":  p["xid"]  for p in pred_records},
        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_bag":  p["bag"]  for p in pred_records},
        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_rank": p["rank"] for p in pred_records},
        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_year": p["year"] for p in pred_records},
        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_yt":   p["yt"]   for p in pred_records},
        **{f"{p['stream']}_{p['policy']}_{p['test_year']}_yp":   p["yp"]   for p in pred_records},
    )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors-dir", default="/fs1/home/atikul/Phenology/cache/tensors")
    ap.add_argument("--output-dir",  default="/fs1/home/atikul/Phenology/phase3_outputs/lstm")
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2017, 2018, 2019, 2020, 2021])
    ap.add_argument("--epochs",     type=int, default=12)
    ap.add_argument("--hidden",     type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--streams",    nargs="+", default=None)
    ap.add_argument("--policies",   nargs="+", default=None)
    args = ap.parse_args()
    main(Path(args.tensors_dir), Path(args.output_dir),
         years=args.years, epochs=args.epochs, hidden=args.hidden,
         batch_size=args.batch_size, streams=args.streams, policies=args.policies)

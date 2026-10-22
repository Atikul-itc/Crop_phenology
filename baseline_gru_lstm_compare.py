"""
LSTM variant of the multi-stream sequence encoder.

The original `phase3_baselines.baseline_lstm` uses `nn.GRU` despite the
class name. This module provides a true LSTM counterpart so we can
compare GRU and LSTM performance on the same data and report which one
the paper reflects.

Conventions:
- StreamEncoderLSTM is a drop-in replacement for StreamEncoder, with the
  same interface (forward takes (x, m) and returns the hidden sequence).
- MultiStreamModel is a generalised version of MultiStreamLSTM that
  accepts an `encoder_cls` argument, so we can instantiate either the
  GRU or the LSTM variant from the same class definition.
- For backward compatibility, MultiStreamGRU and MultiStreamLSTMReal are
  thin aliases.
- soft_label_loss and predict_argmax from baseline_lstm are reused
  unchanged (they operate on logits, not on architecture internals).

Usage:
    from phase3_baselines.baseline_gru_lstm_compare import (
        MultiStreamModel, StreamEncoderGRU, StreamEncoderLSTM, train_one_fold_generic
    )

    # GRU model (equivalent to the existing MultiStreamLSTM)
    model = MultiStreamModel(feature_names, encoder_cls=StreamEncoderGRU,
                              hidden=64, n_phases=7)

    # True LSTM model
    model = MultiStreamModel(feature_names, encoder_cls=StreamEncoderLSTM,
                              hidden=64, n_phases=7)

`train_one_fold_generic` mirrors `train_one_fold` from baseline_lstm but
accepts the encoder class as a parameter.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baseline_lstm import stream_indices, soft_label_loss
from .dataset import make_torch_dataset


# ---------- ENCODERS ----------
class StreamEncoderGRU(nn.Module):
    """GRU-based encoder. Identical to baseline_lstm.StreamEncoder."""
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(input_size=2 * in_dim, hidden_size=hidden,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, x, m):
        inp = torch.cat([x * m, m], dim=-1)
        h, _ = self.rnn(inp)
        return h


class StreamEncoderLSTM(nn.Module):
    """True LSTM-based encoder."""
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size=2 * in_dim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, x, m):
        inp = torch.cat([x * m, m], dim=-1)
        # LSTM returns (output, (h_n, c_n)); we discard the final state tuple
        h, _ = self.rnn(inp)
        return h


# ---------- MODEL (generic over encoder type) ----------
class MultiStreamModel(nn.Module):
    """Multi-stream sequence model with pluggable RNN encoder.

    Identical structure to baseline_lstm.MultiStreamLSTM, but takes the
    encoder class as a constructor argument so we can swap GRU and LSTM
    without duplicating the rest of the model code.
    """
    def __init__(self, feature_names, encoder_cls,
                  hidden: int = 64, num_layers: int = 1,
                  n_phases: int = 7, dropout: float = 0.1):
        super().__init__()
        self.feature_names = feature_names
        self.encoder_cls = encoder_cls
        self.idx = stream_indices(feature_names)
        self.streams = nn.ModuleDict()
        total_dim = 0
        for name, idxs in self.idx.items():
            if len(idxs) == 0:
                continue
            enc = encoder_cls(in_dim=len(idxs), hidden=hidden,
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
                idxs = self.idx[name]
                xs = torch.zeros_like(x[:, :, idxs])
                ms = torch.zeros_like(m[:, :, idxs])
            outs.append(self.streams[name](xs, ms))
        h = torch.cat(outs, dim=-1)
        return self.head(h)


# ---------- TRAINING ----------
def train_one_fold_generic(train_b: dict, test_b: dict,
                             active: tuple,
                             encoder_cls=StreamEncoderGRU,
                             hidden: int = 64, num_layers: int = 1,
                             epochs: int = 12, batch_size: int = 64,
                             lr: float = 1e-3, weight_decay: float = 1e-5,
                             device: torch.device = None,
                             verbose: bool = True,
                             return_model: bool = False):
    """Train one fold with the given encoder class, return predictions.

    Mirrors baseline_lstm.train_one_fold but takes encoder_cls as an
    argument, and optionally returns the trained model so callers can
    run their own inference (for example, to collect per-day logits).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = make_torch_dataset(train_b)
    test_ds  = make_torch_dataset(test_b)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type == "cuda"))

    n_phases = train_b["Y"].shape[2]
    model = MultiStreamModel(
        feature_names=train_b["feature_names"],
        encoder_cls=encoder_cls,
        hidden=hidden, num_layers=num_layers, n_phases=n_phases,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    for ep in range(epochs):
        model.train()
        total, nb = 0.0, 0
        for batch in train_loader:
            x  = batch["X"].to(device, non_blocking=True)
            m  = batch["mask"].to(device, non_blocking=True)
            y  = batch["Y"].to(device, non_blocking=True)
            lm = batch["label_mask"].to(device, non_blocking=True)
            logits = model(x, m, active=active)
            loss = soft_label_loss(logits, y, lm)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        avg = total / max(nb, 1)
        losses.append(avg)
        if verbose:
            print(f"    epoch {ep+1:2d}/{epochs}: train_loss={avg:.4f}")

    # Inference: argmax predictions
    model.eval()
    all_pred, all_truth, all_xid, all_bag, all_rank, all_year = [], [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["X"].to(device)
            m = batch["mask"].to(device)
            y = batch["Y"].to(device)
            lm = batch["label_mask"].to(device)
            logits = model(x, m, active=active)
            pred_day = logits.argmax(dim=1).cpu().numpy()
            truth_day = y.argmax(dim=1).cpu().numpy()
            truth_day[~lm.cpu().numpy()] = -1
            all_pred.append(pred_day); all_truth.append(truth_day)
            all_xid.append(batch["xid"].cpu().numpy())
            all_bag.append(batch["bag_id"].cpu().numpy())
            all_rank.append(batch["rank"].cpu().numpy())
            all_year.append(batch["harvest_year"].cpu().numpy())

    result = {
        "pred":  np.concatenate(all_pred),
        "truth": np.concatenate(all_truth),
        "xid":   np.concatenate(all_xid),
        "bag":   np.concatenate(all_bag),
        "rank":  np.concatenate(all_rank),
        "year":  np.concatenate(all_year),
        "losses": losses,
    }
    if return_model:
        result["model"] = model
        result["test_loader"] = test_loader
    return result


@torch.no_grad()
def collect_logits(model, loader, active, device):
    """Run inference and return full per-day logits, not just argmax."""
    model.eval()
    all_logits, all_yt, all_xid, all_bag, all_rank, all_year = [], [], [], [], [], []
    for batch in loader:
        x  = batch["X"].to(device)
        m  = batch["mask"].to(device)
        y  = batch["Y"].to(device)
        lm = batch["label_mask"].to(device)
        logits = model(x, m, active=active)
        truth_day = y.argmax(dim=1).cpu().numpy()
        truth_day[~lm.cpu().numpy()] = -1
        all_logits.append(logits.cpu().numpy().astype(np.float32))
        all_yt.append(truth_day.astype(np.int32))
        all_xid.append(batch["xid"].cpu().numpy())
        all_bag.append(batch["bag_id"].cpu().numpy())
        all_rank.append(batch["rank"].cpu().numpy())
        all_year.append(batch["harvest_year"].cpu().numpy())
    return {
        "logits": np.concatenate(all_logits, axis=0),
        "yt":     np.concatenate(all_yt, axis=0),
        "xid":    np.concatenate(all_xid).astype(np.int64),
        "bag":    np.concatenate(all_bag).astype(np.int64),
        "rank":   np.concatenate(all_rank).astype(np.int32),
        "year":   np.concatenate(all_year).astype(np.int32),
    }

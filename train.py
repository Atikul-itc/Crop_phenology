"""
Training, prediction, and dual evaluation for Phase 4.

Two evaluation views are produced from a single fold:

  - **Station-level**: argmax of station_logits over T per phase →
    (N_bags, P) predicted days. Compared against the station's reference dates.

  - **Per-field**: for each field, argmax of field_logits over T per phase →
    (N_fields, P) predicted days. Compared against the station-replicated
    reference dates (the same target each field's training sees).

Both go through `phase3_baselines.evaluate.evaluate_predictions`.
"""
from __future__ import annotations
import gc
import time
import numpy as np
import torch

from .bag_dataset import BaggedDataset, build_bags, iter_bag_batches
from .model import MILModel, compute_loss


STREAM_CONFIGS = {
    "S1_only":  ("S1",),
    "S2_only":  ("S2",),
    "Met_only": ("Met",),
    "S1+S2":    ("S1", "S2"),
    "All":      ("S1", "S2", "Met"),
}


def train_mil_fold(train_bags: dict, test_bags: dict, active: tuple,
                    hidden: int = 32, num_layers: int = 1,
                    epochs: int = 15, batch_size: int = 8,
                    lr: float = 1e-3, weight_decay: float = 1e-5,
                    dropout: float = 0.1,
                    phase_resolved_attention: bool = True,
                    field_loss_weight: float = 0.3,
                    use_amp: bool = True,
                    device: torch.device | None = None,
                    verbose: bool = False) -> dict:
    """Train one MIL fold and return predictions + diagnostic arrays."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = BaggedDataset(train_bags)
    ds_te = BaggedDataset(test_bags)
    feature_names = train_bags["feature_names"]
    n_phases = train_bags["Y"].shape[2]

    model = MILModel(feature_names=feature_names,
                       n_phases=n_phases, hidden=hidden, num_layers=num_layers,
                       dropout=dropout,
                       phase_resolved_attention=phase_resolved_attention).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))
    rng = torch.Generator().manual_seed(0)

    losses = []
    for ep in range(epochs):
        model.train()
        tot_loss = 0.0; tot_st = 0.0; tot_fd = 0.0; nb = 0
        for batch in iter_bag_batches(ds_tr, batch_size, True, device, rng):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                out = model(batch, active=active)
                losses_dict = compute_loss(out, batch, field_weight=field_loss_weight)
            scaler.scale(losses_dict["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot_loss += losses_dict["loss"].item()
            tot_st   += losses_dict["L_station"].item()
            tot_fd   += losses_dict["L_field"].item()
            nb += 1
        avg_loss = tot_loss / max(nb, 1)
        losses.append({"epoch": ep + 1, "loss": avg_loss,
                        "L_station": tot_st / max(nb, 1),
                        "L_field":   tot_fd / max(nb, 1)})
        if verbose:
            print(f"    epoch {ep+1:3d}: total={avg_loss:.4f}  "
                   f"st={tot_st/max(nb,1):.4f}  fd={tot_fd/max(nb,1):.4f}")

    # ---------- Predict ----------
    model.eval()
    st_preds, st_truths, st_bagid, st_year = [], [], [], []
    fd_preds, fd_truths, fd_bagid, fd_rank, fd_xid, fd_year = [], [], [], [], [], []
    all_alphas, all_alpha_bagid = [], []

    with torch.no_grad():
        for batch in iter_bag_batches(ds_te, batch_size, False, device):
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                out = model(batch, active=active)

            # Station-level prediction: argmax over T per phase
            st_logits = out["station_logits"]               # (b, T, P)
            st_pred = st_logits.argmax(dim=1).cpu().numpy() # (b, P)
            st_truth = batch["Y"].argmax(dim=1).cpu().numpy()
            st_truth[~batch["label_mask"].cpu().numpy()] = -1
            st_preds.append(st_pred); st_truths.append(st_truth)
            st_bagid.append(batch["idx"].cpu().numpy())     # tmp; resolve to bag_id below
            st_year.append(ds_te.year[batch["idx"]].numpy())

            # Field-level prediction: argmax over T per (B, P)
            fd_logits = out["field_logits"]                 # (b, B, T, P)
            fd_pred = fd_logits.argmax(dim=2).cpu().numpy() # (b, B, P)
            # Replicate station-level truth to each field (same labels across the bag)
            B = fd_logits.shape[1]
            fd_truth = np.repeat(st_truth[:, None, :], B, axis=1)  # (b, B, P)
            # Flatten only valid fields
            bmask = batch["bag_mask"].cpu().numpy()         # (b, B)
            for bi in range(fd_pred.shape[0]):
                for j in range(B):
                    if not bmask[bi, j]:
                        continue
                    fd_preds.append(fd_pred[bi, j])
                    fd_truths.append(fd_truth[bi, j])
                    fd_bagid.append(int(ds_te.bag_id[batch["idx"][bi]].item()))
                    fd_rank.append(int(ds_te.rank[batch["idx"][bi], j].item()))
                    fd_xid.append(int(ds_te.xid[batch["idx"][bi], j].item()))
                    fd_year.append(int(ds_te.year[batch["idx"][bi]].item()))

            # Attention weights for diagnostics: (b, B, P)
            alphas = out["alpha"].cpu().numpy()
            all_alphas.append(alphas)
            all_alpha_bagid.append(ds_te.bag_id[batch["idx"]].numpy())

    # Station-level outputs
    st_pred_all  = np.concatenate(st_preds)
    st_truth_all = np.concatenate(st_truths)
    st_bagid_all = np.concatenate([ds_te.bag_id[i].numpy() for i in st_bagid])
    st_year_all  = np.concatenate(st_year)

    # Field-level outputs
    fd_pred_all  = np.array(fd_preds,  dtype=np.int32) if fd_preds  else np.zeros((0, n_phases), dtype=np.int32)
    fd_truth_all = np.array(fd_truths, dtype=np.int32) if fd_truths else np.zeros((0, n_phases), dtype=np.int32)
    fd_bagid_all = np.array(fd_bagid,  dtype=np.int64)
    fd_rank_all  = np.array(fd_rank,   dtype=np.int32)
    fd_xid_all   = np.array(fd_xid,    dtype=np.int64)
    fd_year_all  = np.array(fd_year,   dtype=np.int32)

    alphas_all       = np.concatenate(all_alphas, axis=0)        # (N_bags, B, P)
    alpha_bagid_all  = np.concatenate(all_alpha_bagid)

    # Cleanup
    del model, opt, scaler, ds_tr, ds_te
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "losses":            losses,
        "station_pred":      st_pred_all,
        "station_truth":     st_truth_all,
        "station_bag_id":    st_bagid_all,
        "station_year":      st_year_all,
        "field_pred":        fd_pred_all,
        "field_truth":       fd_truth_all,
        "field_bag_id":      fd_bagid_all,
        "field_rank":        fd_rank_all,
        "field_xid":         fd_xid_all,
        "field_year":        fd_year_all,
        "alpha":             alphas_all,
        "alpha_bag_id":      alpha_bagid_all,
    }

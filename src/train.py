"""Generic training loop for Phase C neural models on HUPA-UCM.

Model-agnostic: any ``nn.Module`` with signature
``forward(X_dynamic, X_static) -> (B, n_horizons)`` plugs in. Loss-agnostic:
any ``nn.Module`` with signature ``loss_fn(y_pred, y_true) -> scalar``
plugs in.

Per-epoch CSV log -> ``outputs/logs/{run_tag}.csv``.
Best checkpoint (lowest val patient-averaged MAE) -> ``outputs/models/{run_tag}.pt``.
Early stopping with configurable patience. Determinism via ``seed_everything``.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from evaluate import compact_summary, evaluate_model  # noqa: E402


# ---------------------------------------------------------------------------
# Config + reproducibility
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    clip_grad_norm: float | None = 1.0
    seed: int = C.SEED


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def gather_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``(y_true, y_pred, pids)`` over the loader in dataset order.

    Assumes ``shuffle=False`` so the saved ``loader.dataset.pids`` aligns
    row-by-row with the returned arrays.
    """
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device, non_blocking=True)
        x_stat = x_stat.to(device, non_blocking=True)
        y_pred = model(x_dyn, x_stat).detach().cpu().numpy()
        preds.append(y_pred)
        trues.append(y.numpy())
    return (
        np.concatenate(trues, axis=0),
        np.concatenate(preds, axis=0),
        np.asarray(loader.dataset.pids),
    )


def _val_pat_avg_mae(bundle: dict) -> float:
    """Unweighted-mean-across-patients of the mean-over-horizons MAE.

    Patient-averaged MAE is the early-stopping target because long
    participants (HUPA0027 etc.) otherwise dominate pooled MAE — see the
    long-patient-strategy memory.
    """
    pat = bundle["patient_averaged"]
    pat = pat[pat["metric"] == "mae"]
    return float(pat["patient_avg"].mean())


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    loss_fn: nn.Module,
    cfg: TrainConfig,
    run_tag: str,
    logs_dir: Path,
    models_dir: Path,
    verbose: bool = True,
) -> dict:
    """Train ``model`` and return a results dict.

    Returns
    -------
    dict with keys:
        * best_epoch (int)
        * best_val_pat_avg_mae (float)
        * history (list[dict]) — per-epoch metrics
        * final (dict[split, dict])  with ``bundle``, ``compact``,
          and ``predictions`` for val and test under the best checkpoint
        * ckpt_path (str)
        * log_path (str)
    """
    logs_dir = Path(logs_dir)
    models_dir = Path(models_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
        min_lr=cfg.min_lr,
    )

    log_path = logs_dir / f"{run_tag}.csv"
    ckpt_path = models_dir / f"{run_tag}.pt"
    log_fh = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fh)
    log_writer.writerow([
        "epoch", "lr", "train_loss",
        "val_mae_pooled_30m", "val_mae_pooled_60m", "val_mae_pooled_90m",
        "val_mae_pat_avg", "epoch_secs",
    ])

    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []

    try:
        for epoch in range(1, cfg.epochs + 1):
            t_epoch = time.time()
            # ---- train ----
            model.train()
            loss_sum = 0.0
            n_seen = 0
            for x_dyn, x_stat, y in loaders["train"]:
                x_dyn = x_dyn.to(device, non_blocking=True)
                x_stat = x_stat.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(x_dyn, x_stat)
                loss = loss_fn(y_pred, y)
                loss.backward()
                if cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                optimizer.step()
                bs = y.shape[0]
                loss_sum += float(loss.item()) * bs
                n_seen += bs
            train_loss = loss_sum / max(1, n_seen)

            # ---- val ----
            y_true, y_pred, pids = gather_predictions(model, loaders["val"], device)
            bundle = evaluate_model(y_true, y_pred, pids, model_name=run_tag, split_name="val")
            per_h = bundle["per_horizon"]
            mae_per_h = {
                int(h): float(per_h[(per_h["horizon_min"] == h) & (per_h["metric"] == "mae")]["value"].iloc[0])
                for h in C.HORIZON_MINUTES
            }
            val_pat_avg = _val_pat_avg_mae(bundle)

            scheduler.step(val_pat_avg)
            cur_lr = optimizer.param_groups[0]["lr"]
            epoch_secs = time.time() - t_epoch

            history.append({
                "epoch": epoch, "lr": cur_lr, "train_loss": train_loss,
                **{f"val_mae_pooled_{h}m": mae_per_h[h] for h in C.HORIZON_MINUTES},
                "val_mae_pat_avg": val_pat_avg,
                "epoch_secs": epoch_secs,
            })
            log_writer.writerow([
                epoch, f"{cur_lr:.2e}", f"{train_loss:.4f}",
                f"{mae_per_h[30]:.4f}", f"{mae_per_h[60]:.4f}", f"{mae_per_h[90]:.4f}",
                f"{val_pat_avg:.4f}", f"{epoch_secs:.1f}",
            ])
            log_fh.flush()

            if verbose:
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:8.4f}  "
                    f"val_mae(30/60/90)={mae_per_h[30]:6.2f}/{mae_per_h[60]:6.2f}/{mae_per_h[90]:6.2f}  "
                    f"val_pat_avg={val_pat_avg:6.3f}  lr={cur_lr:.2e}  "
                    f"({epoch_secs:5.1f}s)"
                )

            # ---- checkpoint + early-stopping ----
            if val_pat_avg < best_val - 1e-4:
                best_val = val_pat_avg
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_pat_avg_mae": val_pat_avg,
                        "val_mae_per_h": mae_per_h,
                        "config": getattr(model, "config", {}),
                    },
                    ckpt_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.early_stopping_patience:
                    if verbose:
                        print(
                            f"[early-stop] no improvement for "
                            f"{cfg.early_stopping_patience} epochs; "
                            f"best epoch={best_epoch}, best val pat-avg MAE={best_val:.3f}"
                        )
                    break
    finally:
        log_fh.close()

    # ---- final eval under best checkpoint ----
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    final: dict[str, dict] = {}
    for split in ("val", "test"):
        y_true, y_pred, pids = gather_predictions(model, loaders[split], device)
        bundle = evaluate_model(y_true, y_pred, pids, model_name=run_tag, split_name=split)
        final[split] = {
            "bundle": bundle,
            "compact": compact_summary(bundle),
            "predictions": {"y_true": y_true, "y_pred": y_pred, "pids": pids},
        }

    return {
        "best_epoch": best_epoch,
        "best_val_pat_avg_mae": best_val,
        "history": history,
        "final": final,
        "ckpt_path": str(ckpt_path),
        "log_path": str(log_path),
    }

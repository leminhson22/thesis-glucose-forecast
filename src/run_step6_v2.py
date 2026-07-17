"""Train the proposed Hybrid CNN-GRU model with persistence-residual learning.

The public GitHub version keeps the thesis-demo model set focused on six
models. This runner therefore trains the selected proposed model only:
CNN-GRU-Attention with Persistence-Residual Learning.

Usage:
    python src/run_step6_v2.py --epochs 30
    python src/run_step6_v2.py --variant pers_resid --epochs 30

Outputs:
    outputs/logs/step6_hybrid_v2_pers_resid.csv
    outputs/models/step6_hybrid_v2_pers_resid.pt
    outputs/tables/step6_v2_pers_resid_*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from datasets import build_dataloaders, load_npz_splits  # noqa: E402
from evaluate import compact_summary  # noqa: E402
from losses import ZoneWeightedMSE  # noqa: E402
from models import HybridCNNGRUPersResid, count_parameters  # noqa: E402
from train import TrainConfig, get_device, train_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

LOSS_KWARGS = dict(
    w_hypo=2.0,
    w_tir=1.0,
    w_hyper=1.5,
    horizon_weights=(1.5, 1.0, 1.0),
    hypo_under_detect_penalty=2.0,
)
STEP6_EARLY_STOP_PATIENCE = 5
STEP6_MODALITY_DROPOUT_P = 0.30
VARIANT = "pers_resid"


def load_pid_scaler_table(splits: dict) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build per-patient glucose scaler tensors for residual un-scaling."""
    scalers_path = PROJECT_ROOT / C.SCALERS_JSON
    with open(scalers_path, encoding="utf-8") as fh:
        s = json.load(fh)
    per_subject_glu = s["dynamic"]["per_subject"]["glucose"]
    pids = sorted(per_subject_glu.keys())
    pid_lookup = {pid: i for i, pid in enumerate(pids)}
    mean = torch.tensor([per_subject_glu[pid]["mean"] for pid in pids], dtype=torch.float32)
    std = torch.tensor([per_subject_glu[pid]["std"] for pid in pids], dtype=torch.float32)
    return mean, std, pid_lookup


def attach_pid_index_to_static(splits: dict, pid_lookup: dict[str, int]) -> dict:
    """Append a pid-index column to each split's static features."""
    out = {k: splits[k] for k in ("feat_dyn", "feat_stat")}
    for name in ("train", "val", "test"):
        sp = dict(splits[name])
        pids = sp["pids"]
        idx_col = np.array([pid_lookup[p] for p in pids], dtype=np.float32).reshape(-1, 1)
        sp["X_static"] = np.concatenate([sp["X_static"], idx_col], axis=1)
        out[name] = sp
    return out


def make_model(n_dynamic: int, n_static_with_pid: int, feat_dyn: list[str], splits: dict):
    """Construct the selected proposed model."""
    mean, std, _ = load_pid_scaler_table(splits)
    glu_idx = feat_dyn.index("glucose")
    return HybridCNNGRUPersResid(
        n_dynamic=n_dynamic,
        n_static=n_static_with_pid - 1,
        pid_glucose_mean=mean,
        pid_glucose_std=std,
        glucose_dyn_idx=glu_idx,
    )


def save_bundles(bundles: list[dict]) -> pd.DataFrame:
    prefix = "step6_v2_pers_resid"
    name_map = {
        "per_horizon": f"{prefix}_per_horizon.csv",
        "per_zone": f"{prefix}_per_zone.csv",
        "per_patient": f"{prefix}_per_patient.csv",
        "patient_averaged": f"{prefix}_patient_averaged.csv",
        "clarke_eg": f"{prefix}_clarke.csv",
    }
    by_key: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for b in bundles:
        for key, df in b.items():
            by_key[key].append(df)
    for key, frames in by_key.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(TABLES_DIR / name_map[key], index=False)
        print(f"[save] {name_map[key]}  rows={len(out)}")
    compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)
    compact.to_csv(TABLES_DIR / f"{prefix}_summary.csv", index=False)
    print(f"[save] {prefix}_summary.csv  rows={len(compact)}")
    return compact


def main(variant: str, epochs: int) -> int:
    if variant != VARIANT:
        raise ValueError("This repository keeps only --variant pers_resid for the proposed model.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = PROJECT_ROOT / C.SEQUENCES_NPZ
    print(f"[load] {npz_path}")
    splits = load_npz_splits(npz_path)
    sizes = {k: int(splits[k]["y"].shape[0]) for k in ("train", "val", "test")}
    print(f"[load] split counts: {sizes}")

    _, _, pid_lookup = load_pid_scaler_table(splits)
    work_splits = attach_pid_index_to_static(splits, pid_lookup)
    feat_dyn = work_splits["feat_dyn"]
    n_dynamic = len(feat_dyn)
    n_static = work_splits["train"]["X_static"].shape[1]

    model = make_model(n_dynamic, n_static, feat_dyn, splits)
    n_params = count_parameters(model)
    loaders = build_dataloaders(
        work_splits,
        batch_size=128,
        num_workers=0,
        train_modality_dropout_p=STEP6_MODALITY_DROPOUT_P,
        seed=C.SEED,
    )
    loss_fn = ZoneWeightedMSE(**LOSS_KWARGS)
    cfg = TrainConfig(
        epochs=epochs,
        early_stopping_patience=min(STEP6_EARLY_STOP_PATIENCE, max(2, epochs // 3)),
        lr_scheduler_patience=min(3, max(1, epochs // 5)),
    )
    run_tag = "step6_hybrid_v2_pers_resid"

    print(
        f"\n[{run_tag}] {type(model).__name__}  params={n_params:,}  "
        f"device={get_device()}  mod_dropout_p={STEP6_MODALITY_DROPOUT_P}"
    )

    t0 = time.time()
    result = train_model(
        model=model,
        loaders=loaders,
        loss_fn=loss_fn,
        cfg=cfg,
        run_tag=run_tag,
        logs_dir=LOGS_DIR,
        models_dir=MODELS_DIR,
        verbose=True,
    )
    print(
        f"[{run_tag}] done in {time.time() - t0:.1f}s. "
        f"Best epoch={result['best_epoch']}, "
        f"val pat-avg MAE={result['best_val_pat_avg_mae']:.3f}"
    )

    compact = save_bundles(
        [result["final"]["val"]["bundle"], result["final"]["test"]["bundle"]]
    )
    show = [
        "model",
        "split",
        "horizon_min",
        "mae",
        "rmse",
        "mae_pat_avg",
        "clarke_pct_A",
        "clarke_pct_D",
    ]
    show = [c for c in show if c in compact.columns]
    print(f"\n=== {run_tag} compact summary ===")
    print(compact[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    raise SystemExit(main(variant=args.variant, epochs=args.epochs))

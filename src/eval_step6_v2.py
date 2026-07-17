"""Evaluate the selected proposed Hybrid CNN-GRU checkpoint.

This module also exposes ``load_variant_model`` and ``predict_on_split`` for
the Streamlit dashboard and XAI scripts. The public repository keeps only the
selected proposed variant: ``pers_resid``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from datasets import load_npz_splits  # noqa: E402
from evaluate import compact_summary, cg_ega_from_predictions, cg_ega_summary, evaluate_model, zone_of  # noqa: E402
from models import HybridCNNGRUPersResid  # noqa: E402
from run_step6_v2 import attach_pid_index_to_static, load_pid_scaler_table  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
VARIANT = "pers_resid"


def load_variant_model(
    variant: str,
    ckpt_path: Path,
    n_dynamic: int,
    n_static_dataset: int,
    feat_dyn: list[str],
    splits: dict,
):
    """Load the selected proposed checkpoint."""
    if variant != VARIANT:
        raise ValueError("Only the selected proposed variant 'pers_resid' is supported.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    mean, std, _ = load_pid_scaler_table(splits)
    glu_idx = cfg.get("glucose_dyn_idx", feat_dyn.index("glucose"))
    model = HybridCNNGRUPersResid(
        n_dynamic=cfg.get("n_dynamic", n_dynamic),
        n_static=cfg.get("n_static", n_static_dataset - 1),
        pid_glucose_mean=mean,
        pid_glucose_std=std,
        glucose_dyn_idx=glu_idx,
        cnn_channels_per_kernel=cfg.get("cnn_channels_per_kernel", 16),
        cnn_kernels=tuple(cfg.get("cnn_kernels", (3, 5, 7))),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        static_embed_dim=cfg.get("static_embed_dim", 32),
        attn_dim=cfg.get("attn_dim", 48),
        attn_heads=cfg.get("attn_heads", 4),
        head_hidden_dim=cfg.get("head_hidden_dim", 64),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_on_split(model: torch.nn.Module, sp: dict, batch_size: int = 512) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, sp["X_dynamic"].shape[0], batch_size):
            xd = torch.from_numpy(sp["X_dynamic"][i:i + batch_size]).float()
            xs = torch.from_numpy(sp["X_static"][i:i + batch_size]).float()
            preds.append(model(xd, xs).numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def build_predictions_df(split_name: str, sp: dict, y_pred: np.ndarray) -> pd.DataFrame:
    frames = []
    pid = sp["pids"]
    sample_idx = np.arange(sp["X_dynamic"].shape[0], dtype=np.int64)
    for h_idx, h in enumerate(C.HORIZON_MINUTES):
        yt = sp["y"][:, h_idx]
        yp = y_pred[:, h_idx]
        err = yp - yt
        frames.append(pd.DataFrame({
            "model": "step6_hybrid_v2_pers_resid",
            "split": split_name,
            "sample_idx": sample_idx,
            "participant_id": pid,
            "horizon_min": int(h),
            "y_true": yt.astype(np.float32),
            "y_pred": yp.astype(np.float32),
            "abs_err": np.abs(err).astype(np.float32),
            "sq_err": (err ** 2).astype(np.float32),
            "zone": zone_of(yt),
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    npz_path = PROJECT_ROOT / C.SEQUENCES_NPZ
    print(f"[load] {npz_path}")
    splits = load_npz_splits(npz_path)
    feat_dyn = splits["feat_dyn"]

    _, _, pid_lookup = load_pid_scaler_table(splits)
    work_splits = attach_pid_index_to_static(splits, pid_lookup)
    n_static_dataset = work_splits["train"]["X_static"].shape[1]
    ckpt_path = MODELS_DIR / "step6_hybrid_v2_pers_resid.pt"
    model = load_variant_model(
        VARIANT,
        ckpt_path,
        n_dynamic=len(feat_dyn),
        n_static_dataset=n_static_dataset,
        feat_dyn=feat_dyn,
        splits=splits,
    )

    pred_frames: list[pd.DataFrame] = []
    compact_frames: list[pd.DataFrame] = []
    for split_name in ("val", "test"):
        sp = work_splits[split_name]
        y_pred = predict_on_split(model, sp)
        bundle = evaluate_model(
            sp["y"],
            y_pred,
            sp["pids"],
            model_name="step6_hybrid_v2_pers_resid",
            split_name=split_name,
        )
        for key, suffix in {
            "per_horizon": "per_horizon",
            "per_zone": "per_zone",
            "per_patient": "per_patient",
            "patient_averaged": "patient_averaged",
            "clarke_eg": "clarke",
        }.items():
            bundle[key].to_csv(TABLES_DIR / f"step6_v2_pers_resid_{suffix}.csv", index=False)
        compact_frames.append(compact_summary(bundle))
        pred_frames.append(build_predictions_df(split_name, sp, y_pred))

    pred_df = pd.concat(pred_frames, ignore_index=True)
    compact = pd.concat(compact_frames, ignore_index=True)
    compact.to_csv(TABLES_DIR / "step6_v2_pers_resid_summary.csv", index=False)
    pred_df.to_parquet(TABLES_DIR / "step6_v2_predictions.parquet", index=False)
    cg_detail = cg_ega_from_predictions(
        pred_df,
        group_keys=("model", "split", "participant_id", "horizon_min"),
        sort_key="sample_idx",
        rate_lag_steps=3,
        sample_step_min=C.SAMPLING_STEP_MIN,
    )
    cg_overall = cg_ega_summary(cg_detail, group_cols=("model", "split", "horizon_min"), include_zone=False)
    cg_zone = cg_ega_summary(cg_detail, group_cols=("model", "split", "horizon_min"), include_zone=True)
    cg_overall.to_csv(TABLES_DIR / "step6_v2_pers_resid_cg_ega_overall.csv", index=False)
    cg_zone.to_csv(TABLES_DIR / "step6_v2_pers_resid_cg_ega_by_zone.csv", index=False)
    print("[done] evaluated selected proposed checkpoint")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    raise SystemExit(main())

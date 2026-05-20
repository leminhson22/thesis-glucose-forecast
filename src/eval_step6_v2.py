"""Evaluate Step 6 v2 ablation checkpoints with MAE + CG-EGA.

Loads each ``outputs/models/step6_hybrid_v2_<variant>.pt`` that exists,
runs inference on val + test, computes MAE / per-zone / CG-EGA, and
saves a unified comparison table against the §8.6 baseline and the key
Step 5 references (Persistence, HistGB-300, GRU C.2 zwh30a).

Usage::

    python src/eval_step6_v2.py
    python src/eval_step6_v2.py --variants no_moddrop,smaller

Output: outputs/tables/step6_v2_comparison.csv
        outputs/tables/step6_v2_cg_ega_summary.csv (zone × horizon × variant)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from datasets import build_modality_groups, load_npz_splits  # noqa: E402
from evaluate import (  # noqa: E402
    cg_ega_from_predictions,
    cg_ega_summary,
    evaluate_model,
    zone_of,
)
from models import (  # noqa: E402
    HybridCNNGRU,
    HybridCNNGRUNoAttn,
    HybridCNNGRUPersResid,
    HybridModalCNNGRU,
)
from run_step6_v2 import attach_pid_index_to_static, load_pid_scaler_table  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

VARIANTS = ("no_moddrop", "smaller", "no_attn", "modal", "pers_resid")


def load_variant_model(variant: str, ckpt_path: Path, n_dynamic: int, n_static_dataset: int,
                       feat_dyn: list[str], splits: dict):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}

    if variant in ("no_moddrop", "smaller", "baseline"):
        model = HybridCNNGRU(
            n_dynamic=cfg.get("n_dynamic", n_dynamic),
            n_static=cfg.get("n_static", n_static_dataset),
            n_horizons=cfg.get("n_horizons", len(C.HORIZON_MINUTES)),
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
    elif variant == "no_attn":
        model = HybridCNNGRUNoAttn(
            n_dynamic=cfg.get("n_dynamic", n_dynamic),
            n_static=cfg.get("n_static", n_static_dataset),
            n_horizons=cfg.get("n_horizons", len(C.HORIZON_MINUTES)),
            cnn_channels_per_kernel=cfg.get("cnn_channels_per_kernel", 16),
            cnn_kernels=tuple(cfg.get("cnn_kernels", (3, 5, 7))),
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            static_embed_dim=cfg.get("static_embed_dim", 32),
            head_hidden_dim=cfg.get("head_hidden_dim", 64),
            dropout=cfg.get("dropout", 0.3),
        )
    elif variant == "modal":
        groups = cfg.get("modality_index_groups")
        # cfg stores them as lists; rebuild dict
        if isinstance(groups, dict):
            groups_dict = {k: list(v) for k, v in groups.items()}
        else:
            raise ValueError(f"modal variant missing modality_index_groups in cfg")
        model = HybridModalCNNGRU(
            n_dynamic=cfg.get("n_dynamic", n_dynamic),
            n_static=cfg.get("n_static", n_static_dataset),
            modality_index_groups=groups_dict,
            cnn_channels_per_kernel=cfg.get("cnn_channels_per_kernel", 8),
            cnn_kernels=tuple(cfg.get("cnn_kernels", (3, 5, 7))),
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            static_embed_dim=cfg.get("static_embed_dim", 32),
            attn_dim=cfg.get("attn_dim", 48),
            attn_heads=cfg.get("attn_heads", 4),
            head_hidden_dim=cfg.get("head_hidden_dim", 64),
            dropout=cfg.get("dropout", 0.3),
        )
    elif variant == "pers_resid":
        mean, std, pid_lookup = load_pid_scaler_table(splits)
        glu_idx = cfg.get("glucose_dyn_idx", feat_dyn.index("glucose"))
        model = HybridCNNGRUPersResid(
            n_dynamic=cfg.get("n_dynamic", n_dynamic),
            n_static=cfg.get("n_static", n_static_dataset - 1),
            pid_glucose_mean=mean, pid_glucose_std=std, glucose_dyn_idx=glu_idx,
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
    else:
        raise ValueError(f"unknown variant: {variant}")

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


def build_predictions_df(variant: str, split_name: str, sp: dict, y_pred: np.ndarray) -> pd.DataFrame:
    """Tall frame in the same schema as all_models_predictions.parquet."""
    frames = []
    pid = sp["pids"]
    sample_idx = np.arange(sp["X_dynamic"].shape[0], dtype=np.int64)
    for h_idx, h in enumerate(C.HORIZON_MINUTES):
        yt = sp["y"][:, h_idx]
        yp = y_pred[:, h_idx]
        err = yp - yt
        zones = zone_of(yt)
        frames.append(pd.DataFrame({
            "model": f"step6_v2_{variant}",
            "split": split_name,
            "sample_idx": sample_idx,
            "participant_id": pid,
            "horizon_min": int(h),
            "y_true": yt.astype(np.float32),
            "y_pred": yp.astype(np.float32),
            "abs_err": np.abs(err).astype(np.float32),
            "sq_err": (err ** 2).astype(np.float32),
            "zone": zones,
        }))
    return pd.concat(frames, ignore_index=True)


def main(variants: tuple[str, ...]) -> int:
    npz_path = PROJECT_ROOT / C.SEQUENCES_NPZ
    print(f"[load] {npz_path}")
    splits = load_npz_splits(npz_path)
    feat_dyn = splits["feat_dyn"]
    n_dynamic = len(feat_dyn)

    all_pred_frames: list[pd.DataFrame] = []
    bundle_summary_rows: list[dict] = []
    for variant in variants:
        ckpt_path = MODELS_DIR / f"step6_hybrid_v2_{variant}.pt"
        if not ckpt_path.exists():
            print(f"[skip] {variant}: missing {ckpt_path.name}")
            continue
        print(f"\n[eval] variant={variant}  ckpt={ckpt_path.name}")
        if variant == "pers_resid":
            _, _, pid_lookup = load_pid_scaler_table(splits)
            work_splits = attach_pid_index_to_static(splits, pid_lookup)
        else:
            work_splits = splits
        n_static_dataset = work_splits["train"]["X_static"].shape[1]

        model = load_variant_model(variant, ckpt_path, n_dynamic, n_static_dataset,
                                   feat_dyn, splits)
        for split_name in ("val", "test"):
            sp = work_splits[split_name]
            y_pred = predict_on_split(model, sp)
            bundle = evaluate_model(
                sp["y"], y_pred, sp["pids"],
                model_name=f"step6_v2_{variant}", split_name=split_name,
            )
            per_h = bundle["per_horizon"]
            for h_idx, h in enumerate(C.HORIZON_MINUTES):
                mae = float(per_h[(per_h["horizon_min"] == h) & (per_h["metric"] == "mae")]["value"].iloc[0])
                rmse = float(per_h[(per_h["horizon_min"] == h) & (per_h["metric"] == "rmse")]["value"].iloc[0])
                bundle_summary_rows.append({
                    "variant": variant, "split": split_name, "horizon_min": int(h),
                    "mae": mae, "rmse": rmse,
                })
            # Per-zone
            per_zone = bundle["per_zone"]
            per_zone = per_zone[per_zone["metric"] == "mae"]
            for _, row in per_zone.iterrows():
                bundle_summary_rows.append({
                    "variant": variant, "split": split_name, "horizon_min": int(row["horizon_min"]),
                    "zone_mae_" + str(row["zone"]): float(row["value"]),
                })
            # Save predictions for CG-EGA
            all_pred_frames.append(build_predictions_df(variant, split_name, sp, y_pred))

    if not all_pred_frames:
        print("[done] no checkpoints found.")
        return 1

    pred_df = pd.concat(all_pred_frames, ignore_index=True)
    pred_df.to_parquet(TABLES_DIR / "step6_v2_predictions.parquet", index=False)
    print(f"\n[save] step6_v2_predictions.parquet  rows={len(pred_df):,}")

    print("[compute] CG-EGA on v2 variants")
    cg_detail = cg_ega_from_predictions(
        pred_df,
        group_keys=("model", "split", "participant_id", "horizon_min"),
        sort_key="sample_idx",
        rate_lag_steps=3,
        sample_step_min=C.SAMPLING_STEP_MIN,
    )
    cg_overall = cg_ega_summary(cg_detail, group_cols=("model", "split", "horizon_min"), include_zone=False)
    cg_zone = cg_ega_summary(cg_detail, group_cols=("model", "split", "horizon_min"), include_zone=True)
    cg_overall.to_csv(TABLES_DIR / "step6_v2_cg_ega_overall.csv", index=False)
    cg_zone.to_csv(TABLES_DIR / "step6_v2_cg_ega_by_zone.csv", index=False)
    print(f"[save] step6_v2_cg_ega_overall.csv  rows={len(cg_overall)}")
    print(f"[save] step6_v2_cg_ega_by_zone.csv  rows={len(cg_zone)}")

    # Headline printouts on test
    print("\n=== Test pooled MAE by variant (lower is better) ===")
    summary_df = pd.DataFrame([r for r in bundle_summary_rows if 'mae' in r and 'split' in r and r['split'] == 'test' and 'rmse' in r])
    if not summary_df.empty:
        pivot = summary_df.pivot(index='variant', columns='horizon_min', values='mae').round(2)
        pivot.columns = [f"mae_{c}m" for c in pivot.columns]
        print(pivot.to_string())

    print("\n=== Test CG-EGA AP%/EP% by variant ===")
    test_cg = cg_overall[cg_overall['split'] == 'test'].copy()
    ap_pivot = test_cg.pivot(index='model', columns='horizon_min', values='AP_pct').round(2)
    ep_pivot = test_cg.pivot(index='model', columns='horizon_min', values='EP_pct').round(2)
    ap_pivot.columns = [f"AP_{c}m" for c in ap_pivot.columns]
    ep_pivot.columns = [f"EP_{c}m" for c in ep_pivot.columns]
    print("AP%:")
    print(ap_pivot.to_string())
    print("\nEP%:")
    print(ep_pivot.to_string())

    print("\n=== Test CG-EGA hypo zone (clinically critical) ===")
    test_hypo = cg_zone[(cg_zone['split'] == 'test') & (cg_zone['glycaemic_zone'] == 'hypo')].copy()
    hypo_pivot = test_hypo.pivot(index='model', columns='horizon_min', values='EP_pct').round(2)
    hypo_pivot.columns = [f"hypo_EP_{c}m" for c in hypo_pivot.columns]
    print(hypo_pivot.to_string())

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated list; default: all known variants")
    args = ap.parse_args()
    vs = tuple(s.strip() for s in args.variants.split(",") if s.strip())
    raise SystemExit(main(vs))

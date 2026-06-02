"""Clinical Step 6 v2 runner.

Adds two thesis-facing improvements on top of the existing
CNN-GRU-Attention + Persistence-Residual model:

1. validation checkpoint selection by CG-EGA score;
2. optional clinical loss plus balanced hypoglycaemia sampling.

The original run_step6_v2.py remains unchanged for reproducibility.
"""
from __future__ import annotations

import argparse
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
from datasets import build_dataloaders, load_npz_splits  # noqa: E402
from evaluate import cg_ega_from_predictions, cg_ega_summary, compact_summary  # noqa: E402
from losses import ClinicalZoneRateLoss, ZoneWeightedMSE  # noqa: E402
from models import count_parameters  # noqa: E402
from run_step6_v2 import (  # noqa: E402
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    STEP6_MODALITY_DROPOUT_P,
    attach_pid_index_to_static,
    load_pid_scaler_table,
    make_model,
    save_bundles,
)
from train import TrainConfig, _prediction_frame, train_model  # noqa: E402

BASE_LOSS_KWARGS = dict(
    w_hypo=2.0,
    w_tir=1.0,
    w_hyper=1.5,
    horizon_weights=(1.5, 1.0, 1.0),
    hypo_under_detect_penalty=2.0,
)

CLINICAL_LOSS_KWARGS = dict(
    w_hypo=3.0,
    w_tir=1.0,
    w_hyper=1.25,
    horizon_weights=(1.0, 1.25, 1.6),
    hypo_under_detect_penalty=3.0,
    missed_hypo_penalty=0.8,
    missed_hyper_penalty=0.15,
    threshold_margin=5.0,
    direction_penalty=0.08,
    direction_delta_threshold=10.0,
    direction_margin=2.0,
)


def make_clinical_sample_weights(y: np.ndarray) -> np.ndarray:
    """Replacement-sampling weights for rare and near-risk glycaemic states."""
    y = np.asarray(y, dtype=float)
    any_hypo = (y < C.GLUCOSE_HYPO_THRESHOLD).any(axis=1)
    near_hypo = ((y >= C.GLUCOSE_HYPO_THRESHOLD) & (y < 80.0)).any(axis=1)
    any_hyper = (y > C.GLUCOSE_HYPER_THRESHOLD).any(axis=1)
    weights = np.ones(y.shape[0], dtype=np.float64)
    weights[any_hyper] = 1.25
    weights[near_hypo] = np.maximum(weights[near_hypo], 2.0)
    weights[any_hypo] = 5.5
    return weights


def save_prediction_and_cg_ega(result: dict, model_name: str, variant_slug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for split in ("val", "test"):
        pred = result["final"][split]["predictions"]
        frames.append(_prediction_frame(
            pred["y_true"],
            pred["y_pred"],
            pred["pids"],
            model_name=model_name,
            split_name=split,
        ))
    pred_df = pd.concat(frames, ignore_index=True)
    pred_path = TABLES_DIR / f"step6_v2_{variant_slug}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"[save] {pred_path.name} rows={len(pred_df)}")

    cg = cg_ega_from_predictions(pred_df)
    overall = cg_ega_summary(cg, include_zone=False)
    by_zone = cg_ega_summary(cg, include_zone=True)
    overall_path = TABLES_DIR / f"step6_v2_{variant_slug}_cg_ega_overall.csv"
    by_zone_path = TABLES_DIR / f"step6_v2_{variant_slug}_cg_ega_by_zone.csv"
    overall.to_csv(overall_path, index=False)
    by_zone.to_csv(by_zone_path, index=False)
    print(f"[save] {overall_path.name} rows={len(overall)}")
    print(f"[save] {by_zone_path.name} rows={len(by_zone)}")
    return overall, by_zone


def save_six_model_comparison(new_overall: pd.DataFrame, variant_slug: str) -> None:
    rows = []
    base_path = TABLES_DIR / "cg_ega_summary_overall.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        keep = ["persistence", "ridge_a0.1", "rf_n300", "gbm_n300", "gru_c2_zwh30a"]
        rows.append(base[base["model"].isin(keep)])
    old_prop = TABLES_DIR / "step6_v2_cg_ega_overall.csv"
    if old_prop.exists():
        v2 = pd.read_csv(old_prop)
        rows.append(v2[v2["model"].isin(["step6_v2_pers_resid"])])
    rows.append(new_overall)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["split"] == "test"].copy()
    out_path = TABLES_DIR / f"step6_v2_{variant_slug}_cg_ega_comparison_6models.csv"
    out.to_csv(out_path, index=False)
    print(f"[save] {out_path.name} rows={len(out)}")


def run(mode: str, epochs: int, batch_size: int) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_npz_splits(PROJECT_ROOT / C.SEQUENCES_NPZ)
    mean, std, pid_lookup = load_pid_scaler_table(splits)
    work_splits = attach_pid_index_to_static(splits, pid_lookup)
    feat_dyn = work_splits["feat_dyn"]
    n_dynamic = len(feat_dyn)
    n_static = work_splits["train"]["X_static"].shape[1]
    glu_idx = feat_dyn.index("glucose")

    model, mod_p = make_model(
        variant="pers_resid",
        n_dynamic=n_dynamic,
        n_static=n_static,
        feat_dyn=feat_dyn,
        splits=work_splits,
    )

    sample_weights = None
    if mode == "clinical":
        sample_weights = make_clinical_sample_weights(work_splits["train"]["y"])
        loss_fn = ClinicalZoneRateLoss(
            **CLINICAL_LOSS_KWARGS,
            pid_glucose_mean=mean,
            pid_glucose_std=std,
            glucose_dyn_idx=glu_idx,
        )
        variant_slug = "pers_resid_clinical"
    elif mode == "cgega_ckpt":
        loss_fn = ZoneWeightedMSE(**BASE_LOSS_KWARGS)
        variant_slug = "pers_resid_cgega_ckpt"
    else:
        raise ValueError("mode must be 'cgega_ckpt' or 'clinical'")

    loaders = build_dataloaders(
        work_splits,
        batch_size=batch_size,
        num_workers=0,
        train_modality_dropout_p=float(mod_p),
        train_sample_weights=sample_weights,
        seed=C.SEED,
    )

    cfg = TrainConfig(
        epochs=epochs,
        early_stopping_patience=min(6, max(2, epochs // 3)),
        lr_scheduler_patience=min(3, max(1, epochs // 5)),
        checkpoint_metric="cg_ega",
        cg_ega_horizon_weights=(1.0, 1.2, 1.5),
        cg_ega_hypo_ep_weight=2.0,
        cg_ega_hyper_ep_weight=0.5,
        cg_ega_ap_reward=0.2,
    )
    run_tag = f"step6_v2_{variant_slug}"

    print(
        f"[{run_tag}] {type(model).__name__} params={count_parameters(model):,} "
        f"mode={mode} mod_dropout_p={mod_p} batch_size={batch_size}"
    )
    if sample_weights is not None:
        q = np.quantile(sample_weights, [0.0, 0.5, 0.9, 1.0])
        print(f"[sampler] weight quantiles min/median/p90/max={q.tolist()}")
    print(f"[loss] {loss_fn}")

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
        f"[{run_tag}] done in {time.time()-t0:.1f}s. "
        f"best_epoch={result['best_epoch']} "
        f"best_{result['checkpoint_metric']}={result['best_checkpoint_score']:.3f} "
        f"val_pat_avg_mae_at_best={result['best_val_pat_avg_mae']:.3f}"
    )

    compact = save_bundles(
        [result["final"]["val"]["bundle"], result["final"]["test"]["bundle"]],
        variant=variant_slug,
    )
    overall, _ = save_prediction_and_cg_ega(result, model_name=run_tag, variant_slug=variant_slug)
    save_six_model_comparison(overall, variant_slug=variant_slug)

    show = ["model", "split", "horizon_min", "mae", "rmse", "mae_pat_avg", "clarke_pct_A", "clarke_pct_D"]
    show = [c for c in show if c in compact.columns]
    print(compact[show].to_string(index=False))
    print(overall[overall["split"] == "test"].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cgega_ckpt", "clinical"), default="clinical")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    run(mode=args.mode, epochs=args.epochs, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

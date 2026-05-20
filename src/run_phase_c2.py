"""Step 5 Phase C.2 — Zone-weighted / asymmetric loss on the GRU baseline.

Closes Step 5 of SKILL.md by addressing the C.1 30 min hypo bottleneck via
loss-function intervention only. Architecture (RecurrentRegressor with
``rnn_type='gru'``), feature set, split, and hyperparameters are unchanged
from C.1 — the only variables are the loss class and two regularisation
knobs that C.1 retrospectively recommended: ``dropout=0.3`` (was 0.2) and
``early_stopping_patience=5`` (was 10) because C.1 overfit at epoch 2-3.

Configurations:

* ``gru_c2_zw``       — zone-weighted MSE (w_hypo=2.0, w_tir=1.0, w_hyper=1.5).
* ``gru_c2_zwh30``    — zw + 30 min horizon up-weight (1.5x at 30, 1.0 at 60/90).
* ``gru_c2_zwh30a``   — zwh30 + hypo under-detect asymmetric penalty (2.0x).

Each variant is trained from scratch with the same seed so the only source
of variance vs. ``gru_phase_c1`` is the loss formulation. Comparisons in
the printed summary are computed against GBM-300 (best baseline pooled) and
against ``gru_phase_c1`` for the loss-only ablation.

Usage::

    python src/run_phase_c2.py                       # all 3 variants, full
    python src/run_phase_c2.py --debug               # 5k/2k/2k, 2 epochs
    python src/run_phase_c2.py --variant zw          # one variant only
    python src/run_phase_c2.py --variant zwh30a
    python src/run_phase_c2.py --epochs 50           # override epoch cap

Outputs:

    outputs/logs/gru_c2_{variant}.csv
    outputs/models/gru_c2_{variant}.pt
    outputs/tables/phase_c2_summary.csv
    outputs/tables/phase_c2_per_horizon.csv
    outputs/tables/phase_c2_per_zone.csv
    outputs/tables/phase_c2_per_patient.csv
    outputs/tables/phase_c2_patient_averaged.csv
    outputs/tables/phase_c2_clarke.csv
    outputs/tables/phase_c2_vs_c1.csv               # delta vs gru_phase_c1
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from datasets import build_dataloaders, load_npz_splits  # noqa: E402
from evaluate import compact_summary  # noqa: E402
from losses import ZoneWeightedMSE  # noqa: E402
from models import count_parameters, gru_regressor  # noqa: E402
from train import TrainConfig, get_device, train_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Frozen baselines for the end-of-run comparison printout.
# Source: outputs/tables/phase_b_summary.csv and phase_c1_summary.csv (2026-05-19).
GBM_300_TEST_MAE = {30: 10.40, 60: 19.77, 90: 26.27}
GRU_C1_TEST_MAE = {30: 15.76, 60: 21.42, 90: 26.75}

# Phase C.2 retunes: tighter early-stopping + heavier dropout because C.1
# overfit at epoch 2-3 with patience=10/dropout=0.2.
C2_DROPOUT = 0.3
C2_EARLY_STOP_PATIENCE = 5

VARIANT_CONFIGS: dict[str, dict] = {
    "zw": dict(
        loss_kwargs=dict(w_hypo=2.0, w_tir=1.0, w_hyper=1.5),
        description="zone-weighted MSE (w_hypo=2.0, w_hyper=1.5)",
    ),
    "zwh30": dict(
        loss_kwargs=dict(
            w_hypo=2.0, w_tir=1.0, w_hyper=1.5,
            horizon_weights=(1.5, 1.0, 1.0),
        ),
        description="zw + 30m horizon up-weight (1.5x at 30m)",
    ),
    "zwh30a": dict(
        loss_kwargs=dict(
            w_hypo=2.0, w_tir=1.0, w_hyper=1.5,
            horizon_weights=(1.5, 1.0, 1.0),
            hypo_under_detect_penalty=2.0,
        ),
        description="zwh30 + hypo under-detect asymmetric penalty (2.0x)",
    ),
}


def maybe_subsample_splits(splits: dict, n_train: int, n_eval: int, seed: int = C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: splits[k] for k in ("feat_dyn", "feat_stat")}
    for name, cap in (("train", n_train), ("val", n_eval), ("test", n_eval)):
        sp = splits[name]
        n = sp["y"].shape[0]
        if cap <= 0 or n <= cap:
            out[name] = sp
            continue
        idx = np.sort(rng.choice(n, size=cap, replace=False))
        out[name] = {k: v[idx] for k, v in sp.items()}
    return out


def save_bundles(bundles: list[dict]) -> pd.DataFrame:
    name_map = {
        "per_horizon": "phase_c2_per_horizon.csv",
        "per_zone": "phase_c2_per_zone.csv",
        "per_patient": "phase_c2_per_patient.csv",
        "patient_averaged": "phase_c2_patient_averaged.csv",
        "clarke_eg": "phase_c2_clarke.csv",
    }
    by_key: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for b in bundles:
        for key, df in b.items():
            by_key[key].append(df)
    for key, frames in by_key.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(TABLES_DIR / name_map[key], index=False)
        print(f"[save] {name_map[key]:38s} rows={len(out)}")
    compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)
    compact.to_csv(TABLES_DIR / "phase_c2_summary.csv", index=False)
    print(f"[save] phase_c2_summary.csv                rows={len(compact)}")
    return compact


def train_one(
    variant: str,
    splits: dict,
    epochs: int,
    batch_size: int,
    debug: bool,
) -> tuple[dict, dict]:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"unknown variant {variant!r}; pick from {tuple(VARIANT_CONFIGS)}")
    cfg_entry = VARIANT_CONFIGS[variant]

    feat_dyn = splits["feat_dyn"]
    feat_stat = splits["feat_stat"]
    n_dynamic = len(feat_dyn)
    n_static = len(feat_stat)
    loaders = build_dataloaders(splits, batch_size=batch_size, num_workers=0, seed=C.SEED)
    model = gru_regressor(n_dynamic=n_dynamic, n_static=n_static, dropout=C2_DROPOUT)
    n_params = count_parameters(model)

    loss_fn = ZoneWeightedMSE(**cfg_entry["loss_kwargs"])

    cfg = TrainConfig(
        epochs=epochs,
        early_stopping_patience=min(C2_EARLY_STOP_PATIENCE, max(2, epochs // 3)),
        lr_scheduler_patience=min(3, max(1, epochs // 5)),
    )

    run_tag = f"gru_c2_{variant}{'_debug' if debug else ''}"
    print(
        f"\n[gru_c2_{variant}] {cfg_entry['description']}"
        f"\n  loss={loss_fn.extra_repr()}"
        f"\n  dropout={C2_DROPOUT}, patience={cfg.early_stopping_patience}, "
        f"epochs={epochs}, batch={batch_size}, device={get_device()}, params={n_params:,}"
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
        f"[gru_c2_{variant}] done in {time.time() - t0:.1f}s. "
        f"Best epoch={result['best_epoch']}, "
        f"val pat-avg MAE={result['best_val_pat_avg_mae']:.3f}"
    )
    return result["final"]["val"]["bundle"], result["final"]["test"]["bundle"]


def build_vs_c1_table(compact: pd.DataFrame) -> pd.DataFrame:
    test = compact[compact["split"] == "test"].copy()
    rows = []
    for _, row in test.iterrows():
        h = int(row["horizon_min"])
        mae = float(row["mae"])
        delta_c1 = mae - GRU_C1_TEST_MAE[h]
        delta_gbm = mae - GBM_300_TEST_MAE[h]
        rows.append({
            "model": row["model"],
            "horizon_min": h,
            "test_mae": mae,
            "gru_c1_mae": GRU_C1_TEST_MAE[h],
            "delta_vs_gru_c1": delta_c1,
            "pct_vs_gru_c1": 100.0 * delta_c1 / GRU_C1_TEST_MAE[h],
            "gbm_300_mae": GBM_300_TEST_MAE[h],
            "delta_vs_gbm_300": delta_gbm,
            "pct_vs_gbm_300": 100.0 * delta_gbm / GBM_300_TEST_MAE[h],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "phase_c2_vs_c1.csv", index=False)
    print(f"[save] phase_c2_vs_c1.csv                  rows={len(out)}")
    return out


def main(debug: bool, variants: tuple[str, ...], epochs_override: int | None) -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = PROJECT_ROOT / C.SEQUENCES_NPZ
    print(f"[load] {npz_path}")
    splits = load_npz_splits(npz_path)
    sizes = {k: int(splits[k]["y"].shape[0]) for k in ("train", "val", "test")}
    print(
        f"[load] split counts: {sizes}  "
        f"feat_dyn={len(splits['feat_dyn'])}, feat_stat={len(splits['feat_stat'])}"
    )

    if debug:
        print("[debug] subsample train=5000, val=2000, test=2000")
        splits = maybe_subsample_splits(splits, n_train=5000, n_eval=2000)
        epochs = epochs_override or 2
        batch_size = 64
    else:
        epochs = epochs_override or 30
        batch_size = 128

    bundles: list[dict] = []
    t_total = time.time()
    for variant in variants:
        val_b, test_b = train_one(variant, splits, epochs, batch_size, debug)
        bundles.extend([val_b, test_b])

    if not bundles:
        print("[warn] no model trained")
        return 1

    print("\n[summary] writing aggregated tables")
    compact = save_bundles(bundles)

    print("\n========== COMPACT SUMMARY (mg/dL) ==========")
    show = [
        "model", "split", "horizon_min", "mae", "rmse",
        "mae_pat_avg", "rmse_pat_avg",
        "clarke_pct_A", "clarke_pct_D",
    ]
    show = [c for c in show if c in compact.columns]
    print(compact[show].to_string(index=False))

    print("\n========== vs GRU-C1 and GBM-300 (test pooled MAE, mg/dL) ==========")
    vs_table = build_vs_c1_table(compact)
    show_cols = ["model", "horizon_min", "test_mae", "delta_vs_gru_c1",
                 "pct_vs_gru_c1", "delta_vs_gbm_300", "pct_vs_gbm_300"]
    print(vs_table[show_cols].to_string(index=False))

    print(f"\n[done] elapsed = {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--variant",
        choices=("zw", "zwh30", "zwh30a", "all"),
        default="all",
        help="which C.2 loss variant to train (default: all three)",
    )
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    variants = tuple(VARIANT_CONFIGS) if args.variant == "all" else (args.variant,)
    raise SystemExit(main(debug=args.debug, variants=variants, epochs_override=args.epochs))

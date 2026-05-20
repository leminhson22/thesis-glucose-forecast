"""Step 6 — Hybrid CNN-GRU with cross-attention modality fusion (local CLI).

Proposed thesis model per `reports/model_choice_rationale.md`. Architecture
in `src/models.py::HybridCNNGRU`:

    multi-kernel 1D-CNN (k=3,5,7) ──► GRU (h=64, L=2)
                                       │
       static MLP ──► cross-attn (Q = static, KV = GRU sequence)
                                       │
            concat(last_h, attended, static_emb) ──► head ──► (B, 3)

Inputs are the same 17-dynamic + 16-static features the Phase A/B/C
baselines consumed, so the comparison stays apples-to-apples. The loss
is the winning Phase C.2 configuration (`gru_c2_zwh30a`):
``ZoneWeightedMSE(w_hypo=2.0, w_hyper=1.5, horizon_weights=(1.5,1.0,1.0),
hypo_under_detect_penalty=2.0)``. Training enables 30 % modality dropout
per `[[deployment-tier-strategy]]`.

The Step 6 falsification criteria (`reports/report.md` §8.5) are:

1. Pooled MAE approaches HistGB-300 (10.40 / 19.77 / 26.27 mg/dL).
2. Long-horizon hypo and hyper MAE are at most as bad as `gru_c2_zwh30a`
   (17.56 / 25.71 mg/dL at 60 / 90 min hypo; 28.47 / 39.20 mg/dL at hyper).
3. Clarke Zone D share is at most that of HistGB-300 at every horizon
   (1.94 / 6.41 / 8.44 %).

Usage::

    python src/run_step6.py                       # full run, default config
    python src/run_step6.py --debug               # 5k/2k/2k, 2 epochs
    python src/run_step6.py --epochs 40           # override epoch budget
    python src/run_step6.py --no-modality-dropout # ablation: dropout off
    python src/run_step6.py --no-asym             # ablation: no asym penalty

Outputs:

    outputs/logs/step6_hybrid.csv
    outputs/models/step6_hybrid.pt
    outputs/tables/step6_summary.csv
    outputs/tables/step6_per_horizon.csv
    outputs/tables/step6_per_zone.csv
    outputs/tables/step6_per_patient.csv
    outputs/tables/step6_patient_averaged.csv
    outputs/tables/step6_clarke.csv
    outputs/tables/step6_vs_baselines.csv         # delta vs key references
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
from models import HYBRID_DEFAULTS, count_parameters, hybrid_cnn_gru  # noqa: E402
from train import TrainConfig, get_device, train_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Frozen reference numbers for the end-of-run comparison printout.
# Source: outputs/tables/{phase_b,phase_c1,phase_c2,all_models}_*.csv (2026-05-19).
REFERENCE_TEST_MAE = {
    "persistence":   {30: 13.52, 60: 22.92, 90: 29.85},
    "gbm_300":       {30: 10.40, 60: 19.77, 90: 26.27},
    "gru_c2_zwh30a": {30: 15.51, 60: 21.43, 90: 27.00},
}
REFERENCE_HYPO_MAE = {
    "persistence":   {30: 9.06,  60: 18.34, 90: 27.26},
    "gbm_300":       {30: 10.42, 60: 26.75, 90: 40.72},
    "gru_c2_zwh30a": {30: 12.38, 60: 17.56, 90: 25.71},
}

# Winning C.2 loss config, inherited unchanged by Step 6.
LOSS_KWARGS_DEFAULT = dict(
    w_hypo=2.0, w_tir=1.0, w_hyper=1.5,
    horizon_weights=(1.5, 1.0, 1.0),
    hypo_under_detect_penalty=2.0,
)
LOSS_KWARGS_NO_ASYM = {**LOSS_KWARGS_DEFAULT, "hypo_under_detect_penalty": 1.0}

STEP6_EARLY_STOP_PATIENCE = 5
STEP6_MODALITY_DROPOUT_P = 0.30


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


def save_bundles(bundles: list[dict], run_tag: str) -> pd.DataFrame:
    name_map = {
        "per_horizon": "step6_per_horizon.csv",
        "per_zone": "step6_per_zone.csv",
        "per_patient": "step6_per_patient.csv",
        "patient_averaged": "step6_patient_averaged.csv",
        "clarke_eg": "step6_clarke.csv",
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
    compact.to_csv(TABLES_DIR / "step6_summary.csv", index=False)
    print(f"[save] step6_summary.csv                  rows={len(compact)}")
    return compact


def build_vs_baselines_table(compact: pd.DataFrame, run_tag: str) -> pd.DataFrame:
    """Delta vs persistence / gbm_300 / gru_c2_zwh30a for the falsification check."""
    # Also pull per-zone hypo from step6_per_zone.csv
    per_zone = pd.read_csv(TABLES_DIR / "step6_per_zone.csv")
    pz_test_hypo = per_zone[
        (per_zone["split"] == "test") & (per_zone["metric"] == "mae")
        & (per_zone["zone"] == "hypo")
    ].set_index("horizon_min")["value"].to_dict()

    test = compact[compact["split"] == "test"].copy()
    rows = []
    for _, row in test.iterrows():
        h = int(row["horizon_min"])
        mae = float(row["mae"])
        hypo_mae = float(pz_test_hypo.get(h, float("nan")))
        rows.append({
            "model": row["model"],
            "horizon_min": h,
            "test_pooled_mae": mae,
            "test_hypo_mae": hypo_mae,
            "delta_pooled_vs_gbm_300": mae - REFERENCE_TEST_MAE["gbm_300"][h],
            "delta_pooled_vs_zwh30a": mae - REFERENCE_TEST_MAE["gru_c2_zwh30a"][h],
            "delta_hypo_vs_persistence": hypo_mae - REFERENCE_HYPO_MAE["persistence"][h],
            "delta_hypo_vs_gbm_300": hypo_mae - REFERENCE_HYPO_MAE["gbm_300"][h],
            "delta_hypo_vs_zwh30a": hypo_mae - REFERENCE_HYPO_MAE["gru_c2_zwh30a"][h],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "step6_vs_baselines.csv", index=False)
    print(f"[save] step6_vs_baselines.csv             rows={len(out)}")
    return out


def train_one(
    run_tag: str,
    splits: dict,
    epochs: int,
    batch_size: int,
    loss_kwargs: dict,
    modality_dropout_p: float,
    debug: bool,
) -> tuple[dict, dict]:
    feat_dyn = splits["feat_dyn"]
    feat_stat = splits["feat_stat"]
    n_dynamic = len(feat_dyn)
    n_static = len(feat_stat)
    loaders = build_dataloaders(
        splits, batch_size=batch_size, num_workers=0,
        train_modality_dropout_p=float(modality_dropout_p),
        seed=C.SEED,
    )
    model = hybrid_cnn_gru(n_dynamic=n_dynamic, n_static=n_static)
    n_params = count_parameters(model)

    loss_fn = ZoneWeightedMSE(**loss_kwargs)

    cfg = TrainConfig(
        epochs=epochs,
        early_stopping_patience=min(STEP6_EARLY_STOP_PATIENCE, max(2, epochs // 3)),
        lr_scheduler_patience=min(3, max(1, epochs // 5)),
    )

    print(
        f"\n[step6_hybrid] HybridCNNGRU"
        f"\n  loss={loss_fn.extra_repr()}"
        f"\n  modality_dropout_p={modality_dropout_p}, "
        f"dropout={HYBRID_DEFAULTS['dropout']}, patience={cfg.early_stopping_patience}"
        f"\n  epochs={epochs}, batch={batch_size}, device={get_device()}, params={n_params:,}"
        f"\n  n_dyn={n_dynamic}, n_stat={n_static}"
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
        f"[step6_hybrid] done in {time.time() - t0:.1f}s. "
        f"Best epoch={result['best_epoch']}, "
        f"val pat-avg MAE={result['best_val_pat_avg_mae']:.3f}"
    )
    return result["final"]["val"]["bundle"], result["final"]["test"]["bundle"]


def main(
    debug: bool,
    epochs_override: int | None,
    no_modality_dropout: bool,
    no_asym: bool,
) -> int:
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

    loss_kwargs = LOSS_KWARGS_NO_ASYM if no_asym else LOSS_KWARGS_DEFAULT
    modality_dropout_p = 0.0 if no_modality_dropout else STEP6_MODALITY_DROPOUT_P

    tag_bits = ["step6_hybrid"]
    if no_modality_dropout:
        tag_bits.append("no_moddrop")
    if no_asym:
        tag_bits.append("no_asym")
    if debug:
        tag_bits.append("debug")
    run_tag = "_".join(tag_bits)

    t_total = time.time()
    val_b, test_b = train_one(
        run_tag, splits, epochs, batch_size,
        loss_kwargs=loss_kwargs,
        modality_dropout_p=modality_dropout_p,
        debug=debug,
    )

    print("\n[summary] writing aggregated tables")
    compact = save_bundles([val_b, test_b], run_tag=run_tag)

    print("\n========== COMPACT SUMMARY (mg/dL) ==========")
    show = [
        "model", "split", "horizon_min", "mae", "rmse",
        "mae_pat_avg", "rmse_pat_avg",
        "clarke_pct_A", "clarke_pct_D",
    ]
    show = [c for c in show if c in compact.columns]
    print(compact[show].to_string(index=False))

    print("\n========== vs key baselines (test, mg/dL) ==========")
    vs = build_vs_baselines_table(compact, run_tag=run_tag)
    print(vs.to_string(index=False))

    # Falsification checks
    print("\n========== Step 6 falsification check ==========")
    test_pooled = vs.set_index("horizon_min")["test_pooled_mae"].to_dict()
    test_hypo = vs.set_index("horizon_min")["test_hypo_mae"].to_dict()
    for h in (30, 60, 90):
        crit1_delta = test_pooled[h] - REFERENCE_TEST_MAE["gbm_300"][h]
        crit2_delta = test_hypo[h] - REFERENCE_HYPO_MAE["gru_c2_zwh30a"][h]
        print(
            f"  {h}m: pooled_vs_gbm_300={crit1_delta:+.2f} (target <=0)  "
            f"hypo_vs_zwh30a={crit2_delta:+.2f} (target <=0)"
        )

    print(f"\n[done] elapsed = {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-modality-dropout", action="store_true",
                    help="ablation: disable training-time modality dropout")
    ap.add_argument("--no-asym", action="store_true",
                    help="ablation: disable hypo under-detect asymmetric penalty")
    args = ap.parse_args()
    raise SystemExit(main(
        debug=args.debug,
        epochs_override=args.epochs,
        no_modality_dropout=args.no_modality_dropout,
        no_asym=args.no_asym,
    ))

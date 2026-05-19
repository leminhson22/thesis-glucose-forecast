"""Step 5 Phase C.1 — LSTM + GRU baselines (local CLI runner).

Trains LSTM and GRU recurrent models on the HUPA-UCM ``X_dynamic`` (B, 24, 17)
plus ``X_static`` (B, 16) input — the same input contract as the Phase A/B
flattened tree baselines, so Phase C.1 numbers are apples-to-apples vs.
Ridge / RF / HistGB.

Phase C.1 uses vanilla ``MultiHorizonMSE`` loss. Phase C.2 will switch to
asymmetric / zone-weighted variants; Phase C.3 adds modality dropout.

Usage:
    python src/run_phase_c1.py                  # both LSTM and GRU, full
    python src/run_phase_c1.py --debug          # 5k/2k/2k subsample, 2 epochs
    python src/run_phase_c1.py --model lstm     # LSTM only
    python src/run_phase_c1.py --model gru      # GRU only
    python src/run_phase_c1.py --epochs 50      # override epoch count

Outputs:
    outputs/logs/{lstm|gru}_phase_c1.csv
    outputs/models/{lstm|gru}_phase_c1.pt
    outputs/tables/phase_c1_summary.csv
    outputs/tables/phase_c1_per_horizon.csv
    outputs/tables/phase_c1_per_zone.csv
    outputs/tables/phase_c1_per_patient.csv
    outputs/tables/phase_c1_patient_averaged.csv
    outputs/tables/phase_c1_clarke.csv
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
from losses import MultiHorizonMSE  # noqa: E402
from models import count_parameters, gru_regressor, lstm_regressor  # noqa: E402
from train import TrainConfig, get_device, train_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Phase B GBM-300 test pooled MAEs, used only for the end-of-run comparison
# printout. Source: outputs/tables/phase_b_summary.csv (frozen 2026-05-19).
GBM_300_TEST_MAE = {30: 10.40, 60: 19.77, 90: 26.27}


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
        "per_horizon": "phase_c1_per_horizon.csv",
        "per_zone": "phase_c1_per_zone.csv",
        "per_patient": "phase_c1_per_patient.csv",
        "patient_averaged": "phase_c1_patient_averaged.csv",
        "clarke_eg": "phase_c1_clarke.csv",
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
    compact.to_csv(TABLES_DIR / "phase_c1_summary.csv", index=False)
    print(f"[save] phase_c1_summary.csv                rows={len(compact)}")
    return compact


def train_one(
    rnn_type: str,
    splits: dict,
    epochs: int,
    batch_size: int,
    debug: bool,
) -> tuple[dict, dict]:
    feat_dyn = splits["feat_dyn"]
    feat_stat = splits["feat_stat"]
    n_dynamic = len(feat_dyn)
    n_static = len(feat_stat)
    loaders = build_dataloaders(splits, batch_size=batch_size, num_workers=0, seed=C.SEED)
    if rnn_type == "lstm":
        model = lstm_regressor(n_dynamic=n_dynamic, n_static=n_static)
    else:
        model = gru_regressor(n_dynamic=n_dynamic, n_static=n_static)
    n_params = count_parameters(model)
    loss_fn = MultiHorizonMSE(reduction="mean")
    cfg = TrainConfig(
        epochs=epochs,
        early_stopping_patience=min(10, max(2, epochs // 3)),
        lr_scheduler_patience=min(5, max(1, epochs // 5)),
    )
    run_tag = f"{rnn_type}_phase_c1{'_debug' if debug else ''}"
    print(
        f"\n[{rnn_type}] start: epochs={epochs}, batch={batch_size}, device={get_device()}, "
        f"params={n_params:,}, n_dyn={n_dynamic}, n_stat={n_static}"
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
        f"[{rnn_type}] done in {time.time() - t0:.1f}s. "
        f"Best epoch={result['best_epoch']}, "
        f"val pat-avg MAE={result['best_val_pat_avg_mae']:.3f}"
    )
    return result["final"]["val"]["bundle"], result["final"]["test"]["bundle"]


def main(debug: bool, model_choice: str, epochs_override: int | None) -> int:
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

    rnn_types = ("lstm", "gru") if model_choice == "both" else (model_choice,)
    bundles: list[dict] = []
    t_total = time.time()
    for rnn_type in rnn_types:
        val_b, test_b = train_one(rnn_type, splits, epochs, batch_size, debug)
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

    print("\n========== vs GBM-300 (test pooled MAE, mg/dL) ==========")
    test_compact = compact[compact["split"] == "test"]
    for _, row in test_compact.iterrows():
        h = int(row["horizon_min"])
        mae = float(row["mae"])
        delta = mae - GBM_300_TEST_MAE[h]
        rel = 100.0 * delta / GBM_300_TEST_MAE[h]
        print(
            f"  {row['model']:30s} {h}m: MAE={mae:6.3f}  "
            f"GBM-300={GBM_300_TEST_MAE[h]:.2f}  "
            f"delta={delta:+.3f} ({rel:+.2f}%)"
        )

    print(f"\n[done] elapsed = {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--model", choices=("lstm", "gru", "both"), default="both")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    raise SystemExit(main(debug=args.debug, model_choice=args.model, epochs_override=args.epochs))

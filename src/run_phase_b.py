"""Step 5 Phase B -- Random Forest baseline runner.

This public repository keeps the six thesis-demo models only:
Persistence, Ridge Regression, Random Forest, LSTM, GRU, and the proposed
Hybrid CNN-GRU. Phase B therefore trains and evaluates Random Forest only.

Usage:
    python src/run_phase_b.py
    python src/run_phase_b.py --debug

Outputs:
    outputs/tables/phase_b_summary.csv
    outputs/tables/phase_b_per_horizon.csv
    outputs/tables/phase_b_per_zone.csv
    outputs/tables/phase_b_per_patient.csv
    outputs/tables/phase_b_patient_averaged.csv
    outputs/tables/phase_b_clarke.csv
    outputs/tables/phase_b_rf_top_importance.csv
    outputs/models/rf_phase_b.joblib
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
from baselines import RandomForestBaseline  # noqa: E402
from evaluate import compact_summary, evaluate_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def load_data() -> dict:
    npz_path = PROJECT_ROOT / C.SEQUENCES_NPZ
    print(f"[load] {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    split = d["split"].astype(str)
    out = {
        "X_dyn": d["X_dynamic"].astype(np.float32),
        "X_stat": d["X_static"].astype(np.float32),
        "y": d["y"].astype(np.float32),
        "pid": d["participant_ids"].astype(str),
        "split": split,
        "feat_dyn": [str(s) for s in d["feature_names_dynamic"]],
        "feat_stat": [str(s) for s in d["feature_names_static"]],
    }
    sizes = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    print(f"[load] split counts: {sizes}  total={sum(sizes.values())}")
    return out


def slice_split(data: dict, name: str) -> dict:
    mask = data["split"] == name
    return {k: data[k][mask] for k in ("X_dyn", "X_stat", "y", "pid")}


def maybe_subsample(sp: dict, n_cap: int, seed: int = C.SEED) -> dict:
    if n_cap <= 0 or len(sp["y"]) <= n_cap:
        return sp
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(sp["y"]), size=n_cap, replace=False))
    return {k: v[idx] for k, v in sp.items()}


def save_bundles(bundles: list[dict]) -> pd.DataFrame:
    name_map = {
        "per_horizon": "phase_b_per_horizon.csv",
        "per_zone": "phase_b_per_zone.csv",
        "per_patient": "phase_b_per_patient.csv",
        "patient_averaged": "phase_b_patient_averaged.csv",
        "clarke_eg": "phase_b_clarke.csv",
    }
    by_key: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for b in bundles:
        for key, df in b.items():
            by_key[key].append(df)
    for key, frames in by_key.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(TABLES_DIR / name_map[key], index=False)
        print(f"[save] {name_map[key]:35s} rows={len(out)}")
    compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)
    compact.to_csv(TABLES_DIR / "phase_b_summary.csv", index=False)
    print(f"[save] phase_b_summary.csv                rows={len(compact)}")
    return compact


def main(debug: bool) -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_data()
    train = slice_split(data, "train")
    val = slice_split(data, "val")
    test = slice_split(data, "test")

    if debug:
        print("[debug] subsampling: train=5000, val=2000, test=2000")
        train = maybe_subsample(train, 5000)
        val = maybe_subsample(val, 2000)
        test = maybe_subsample(test, 2000)
        rf_n_estimators = 50
    else:
        rf_n_estimators = 300

    print(f"\n[rf] fitting n_estimators={rf_n_estimators}, max_depth=25, n_jobs=-1")
    t_rf = time.time()
    rf = RandomForestBaseline(
        n_estimators=rf_n_estimators,
        max_depth=25,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=C.SEED,
    ).fit(
        train["X_dyn"],
        train["X_stat"],
        train["y"],
        feature_names_dynamic=data["feat_dyn"],
        feature_names_static=data["feat_stat"],
    )
    print(f"[rf] fit time = {time.time() - t_rf:.1f}s")

    imp = rf.feature_importance_table(top_k=20)
    imp.insert(0, "model", "rf_n300" if not debug else "rf_n50_debug")
    imp.to_csv(TABLES_DIR / "phase_b_rf_top_importance.csv", index=False)
    print(f"[save] phase_b_rf_top_importance.csv  rows={len(imp)}")

    try:
        import joblib

        joblib.dump(
            {"model": rf.model_, "params": rf.params, "feature_names": rf.feature_names_},
            MODELS_DIR / "rf_phase_b.joblib",
        )
        print("[save] rf_phase_b.joblib")
    except Exception as e:
        print(f"[warn] could not save RF joblib: {e}")

    bundles: list[dict] = []
    rf_tag = f"rf_n{rf_n_estimators}"
    for name, sp in (("val", val), ("test", test)):
        yhat = rf.predict(sp["X_dyn"], sp["X_stat"])
        bundles.append(evaluate_model(sp["y"], yhat, sp["pid"], rf_tag, name))

    print("\n[summary] writing aggregated tables")
    compact = save_bundles(bundles)

    print("\n========== COMPACT SUMMARY (mg/dL) ==========")
    show = [
        "model",
        "split",
        "horizon_min",
        "mae",
        "rmse",
        "mae_pat_avg",
        "rmse_pat_avg",
        "clarke_pct_A",
        "clarke_pct_B",
        "clarke_pct_D",
    ]
    show = [c for c in show if c in compact.columns]
    print(compact[show].to_string(index=False))

    print(f"\n[done] elapsed = {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(debug=args.debug))

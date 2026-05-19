"""Step 5 Phase A — Persistence + Ridge baselines (local CLI runner).

Loads the pre-built ``hupa_5min_sequences.npz`` (with embedded
``participant_ids`` and ``split`` arrays), fits both baselines, evaluates on
VAL and TEST, and writes every artefact to ``outputs/tables/`` and
``outputs/models/``. The notebook ``notebooks/04_model_training.ipynb`` wraps
the same logic with Colab boilerplate; this script is the headless
entrypoint for local runs.

Usage:
    python src/run_phase_a.py            # full run, ~30s on CPU
    python src/run_phase_a.py --debug    # 5k train / 2k val / 2k test, 2 alphas

Outputs (all relative to project root):
    outputs/tables/phase_a_summary.csv               compact one-row-per-(model,split,horizon)
    outputs/tables/phase_a_per_horizon.csv           pooled MAE/RMSE
    outputs/tables/phase_a_per_zone.csv              MAE/RMSE per hypo/tir/hyper
    outputs/tables/phase_a_per_patient.csv           MAE/RMSE per patient
    outputs/tables/phase_a_patient_averaged.csv      mean/sd across patients
    outputs/tables/phase_a_clarke.csv                Clarke EGA zone percentages
    outputs/tables/phase_a_ridge_alpha_tuning.csv    val MAE per candidate alpha
    outputs/tables/phase_a_ridge_top_coefs.csv       top-15 |coef| per horizon
    outputs/models/ridge_phase_a.joblib              fitted Ridge for reuse
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the package importable both as `python src/run_phase_a.py` and `python -m src.run_phase_a`
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from baselines import PersistenceModel, RidgeBaseline, tune_ridge_alpha  # noqa: E402
from evaluate import compact_summary, evaluate_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def load_data() -> dict:
    """Load sequences.npz and split the arrays by the embedded ``split`` mask."""
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
    print(
        f"[load] X_dyn={out['X_dyn'].shape}  X_stat={out['X_stat'].shape}  "
        f"y={out['y'].shape}  features dyn={len(out['feat_dyn'])} stat={len(out['feat_stat'])}"
    )
    return out


def slice_split(data: dict, name: str) -> dict:
    mask = data["split"] == name
    return {
        "X_dyn": data["X_dyn"][mask],
        "X_stat": data["X_stat"][mask],
        "y": data["y"][mask],
        "pid": data["pid"][mask],
    }


def maybe_subsample(split_dict: dict, n_cap: int, seed: int = C.SEED) -> dict:
    if n_cap <= 0 or len(split_dict["y"]) <= n_cap:
        return split_dict
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(split_dict["y"]), size=n_cap, replace=False))
    return {k: v[idx] for k, v in split_dict.items()}


def save_bundles(bundles: list[dict], tag: str) -> None:
    """Concatenate per-frame bundles across (model, split) and save to CSV."""
    by_key: dict[str, list[pd.DataFrame]] = {}
    for b in bundles:
        for key, df in b.items():
            by_key.setdefault(key, []).append(df)
    name_map = {
        "per_horizon": "phase_a_per_horizon.csv",
        "per_zone": "phase_a_per_zone.csv",
        "per_patient": "phase_a_per_patient.csv",
        "patient_averaged": "phase_a_patient_averaged.csv",
        "clarke_eg": "phase_a_clarke.csv",
    }
    for key, frames in by_key.items():
        out = pd.concat(frames, ignore_index=True)
        out_path = TABLES_DIR / name_map[key]
        out.to_csv(out_path, index=False)
        print(f"[save] {out_path.name}  rows={len(out)}")
    # Compact one-row-per-(model, split, horizon)
    compact_frames = [compact_summary(b) for b in bundles]
    compact = pd.concat(compact_frames, ignore_index=True)
    compact_path = TABLES_DIR / "phase_a_summary.csv"
    compact.to_csv(compact_path, index=False)
    print(f"[save] {compact_path.name}  rows={len(compact)}")
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
        print("[debug] subsampling: train=5000, val=2000, test=2000, alphas=(1, 100)")
        train = maybe_subsample(train, 5000)
        val = maybe_subsample(val, 2000)
        test = maybe_subsample(test, 2000)
        alphas = (1.0, 100.0)
    else:
        alphas = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

    bundles: list[dict] = []

    # ---------------- Persistence ----------------
    print("\n[persistence] loading scalers and predicting")
    pers = PersistenceModel.from_scalers_json(
        PROJECT_ROOT / C.SCALERS_JSON,
        glucose_feature_index=data["feat_dyn"].index("glucose"),
    )
    for split_name, sp in (("val", val), ("test", test)):
        y_hat = pers.predict(sp["X_dyn"], sp["pid"], n_horizons=len(C.HORIZON_MINUTES))
        bundle = evaluate_model(sp["y"], y_hat, sp["pid"], "persistence", split_name)
        bundles.append(bundle)

    # ---------------- Ridge ----------------
    print("\n[ridge] tuning alpha on TRAIN -> VAL")
    best_alpha, ridge, alpha_log = tune_ridge_alpha(
        train["X_dyn"], train["X_stat"], train["y"],
        val["X_dyn"], val["X_stat"], val["y"],
        alphas=alphas,
        feature_names_dynamic=data["feat_dyn"],
        feature_names_static=data["feat_stat"],
        verbose=True,
    )
    print(f"[ridge] best alpha = {best_alpha}")
    alpha_log.to_csv(TABLES_DIR / "phase_a_ridge_alpha_tuning.csv", index=False)
    print(f"[save] phase_a_ridge_alpha_tuning.csv  rows={len(alpha_log)}")

    # Top-15 coefs per horizon
    top_coefs = ridge.coef_table(top_k=15)
    top_coefs.insert(0, "best_alpha", best_alpha)
    top_coefs.to_csv(TABLES_DIR / "phase_a_ridge_top_coefs.csv", index=False)
    print(f"[save] phase_a_ridge_top_coefs.csv  rows={len(top_coefs)}")

    # Save fitted Ridge
    try:
        import joblib
        joblib.dump(
            {"model": ridge.model_, "alpha": best_alpha,
             "feature_names": ridge.feature_names_},
            MODELS_DIR / "ridge_phase_a.joblib",
        )
        print(f"[save] ridge_phase_a.joblib")
    except Exception as e:  # joblib optional
        print(f"[warn] could not save joblib: {e}")

    for split_name, sp in (("val", val), ("test", test)):
        y_hat = ridge.predict(sp["X_dyn"], sp["X_stat"])
        bundle = evaluate_model(sp["y"], y_hat, sp["pid"], f"ridge_a{best_alpha:g}", split_name)
        bundles.append(bundle)

    # ---------------- Save & print ----------------
    print("\n[summary] writing aggregated tables")
    compact = save_bundles(bundles, tag="phase_a")

    print("\n========== COMPACT SUMMARY (mg/dL) ==========")
    show_cols = ["model", "split", "horizon_min", "mae", "rmse",
                 "mae_pat_avg", "rmse_pat_avg",
                 "clarke_pct_A", "clarke_pct_B", "clarke_pct_D"]
    show_cols = [c for c in show_cols if c in compact.columns]
    print(compact[show_cols].to_string(index=False))

    print(f"\n[done] elapsed = {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true",
                    help="subsample for fast sanity check (~5s)")
    args = ap.parse_args()
    raise SystemExit(main(debug=args.debug))

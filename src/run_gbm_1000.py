"""Step 5 Phase B — GBM re-tune with max_iter=1000 (ablation before Phase C).

The GBM-300 baseline produced by ``run_phase_b.py`` exhausted its iteration
cap on every horizon (n_iters_used = 300/300/300), so early stopping never
triggered. This script re-runs HistGradientBoosting with ``max_iter = 1000``
keeping every other hyperparameter, the same split, features, seed, and
metric protocol, then writes a side-by-side comparison so we can decide
whether GBM-1000 should replace GBM-300 as the primary tree baseline.

The original Phase B artefacts are NEVER overwritten. The new model is
saved under ``gbm_phase_b_1000.joblib`` and the new metric tables live in
the ``phase_b_gbm1000_*.csv`` namespace.

Usage:
    python src/run_gbm_1000.py
    python src/run_gbm_1000.py --debug   # 5k train / 2k val / 2k test smoke-test

Files written:
    outputs/models/gbm_phase_b_1000.joblib
    outputs/tables/phase_b_gbm1000_summary.csv
    outputs/tables/phase_b_gbm1000_per_horizon.csv
    outputs/tables/phase_b_gbm1000_per_zone.csv
    outputs/tables/phase_b_gbm1000_per_patient.csv
    outputs/tables/phase_b_gbm1000_patient_averaged.csv
    outputs/tables/phase_b_gbm1000_clarke.csv
    outputs/tables/phase_b_gbm1000_n_iters.csv
    outputs/tables/phase_b_gbm_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from baselines import HistGBMBaseline, flatten_window  # noqa: E402
from evaluate import compact_summary, evaluate_model  # noqa: E402

PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


# ---------------------------------------------------------------------------
# Data loading (mirrors run_phase_b.py exactly)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reload the saved GBM-300 so we can re-evaluate it under the identical
# protocol used for GBM-1000 in this run (rather than trusting the older
# phase_b_*.csv values, which were produced separately).
# ---------------------------------------------------------------------------

class _GBM300Reload:
    """Thin wrapper exposing ``.predict(X_dyn, X_stat)`` for the GBM-300 ckpt."""

    def __init__(self, ckpt: dict):
        self.models_ = ckpt["models"]
        self.params = ckpt.get("params", {})
        self.n_iters_used_ = ckpt.get("n_iters_used", [-1, -1, -1])
        self.feature_names_ = ckpt.get("feature_names")

    def predict(self, X_dynamic: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        X = flatten_window(X_dynamic, X_static)
        preds = np.stack([m.predict(X) for m in self.models_], axis=1)
        return preds.astype(np.float32)


# ---------------------------------------------------------------------------
# Bundle saver (separate namespace from Phase B)
# ---------------------------------------------------------------------------

def save_bundles(bundles: list[dict]) -> pd.DataFrame:
    name_map = {
        "per_horizon": "phase_b_gbm1000_per_horizon.csv",
        "per_zone": "phase_b_gbm1000_per_zone.csv",
        "per_patient": "phase_b_gbm1000_per_patient.csv",
        "patient_averaged": "phase_b_gbm1000_patient_averaged.csv",
        "clarke_eg": "phase_b_gbm1000_clarke.csv",
    }
    by_key: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for b in bundles:
        for key, df in b.items():
            by_key[key].append(df)
    for key, frames in by_key.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(TABLES_DIR / name_map[key], index=False)
        print(f"[save] {name_map[key]:42s} rows={len(out)}")
    compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)
    compact.to_csv(TABLES_DIR / "phase_b_gbm1000_summary.csv", index=False)
    print(f"[save] phase_b_gbm1000_summary.csv             rows={len(compact)}")
    return compact


# ---------------------------------------------------------------------------
# Comparison table: per (split, horizon) delta of pooled MAE/RMSE, per-zone
# MAE, patient-averaged MAE, and Clarke %A. Rows are ordered as one block
# per (split, horizon) for readability.
# ---------------------------------------------------------------------------

def build_comparison_table(
    bundles: list[dict],
    tag_300: str,
    tag_1000: str,
) -> pd.DataFrame:
    per_h = pd.concat([b["per_horizon"] for b in bundles], ignore_index=True)
    per_z = pd.concat([b["per_zone"] for b in bundles], ignore_index=True)
    pat = pd.concat([b["patient_averaged"] for b in bundles], ignore_index=True)
    clarke = pd.concat([b["clarke_eg"] for b in bundles], ignore_index=True)

    rows = []
    for split in ("val", "test"):
        for h in C.HORIZON_MINUTES:
            def _ph(model: str, metric: str) -> float:
                m = per_h[(per_h["model"] == model) & (per_h["split"] == split)
                         & (per_h["horizon_min"] == h) & (per_h["metric"] == metric)]
                return float(m["value"].iloc[0]) if len(m) else float("nan")

            def _pz(model: str, zone: str, metric: str = "mae") -> float:
                m = per_z[(per_z["model"] == model) & (per_z["split"] == split)
                         & (per_z["horizon_min"] == h) & (per_z["zone"] == zone)
                         & (per_z["metric"] == metric)]
                return float(m["value"].iloc[0]) if len(m) else float("nan")

            def _pa(model: str, metric: str) -> float:
                m = pat[(pat["model"] == model) & (pat["split"] == split)
                        & (pat["horizon_min"] == h) & (pat["metric"] == metric)]
                return float(m["patient_avg"].iloc[0]) if len(m) else float("nan")

            def _clarke(model: str, zone: str) -> float:
                m = clarke[(clarke["model"] == model) & (clarke["split"] == split)
                           & (clarke["horizon_min"] == h) & (clarke["zone"] == zone)]
                return float(m["pct"].iloc[0]) if len(m) else float("nan")

            mae_300 = _ph(tag_300, "mae")
            mae_1000 = _ph(tag_1000, "mae")
            rmse_300 = _ph(tag_300, "rmse")
            rmse_1000 = _ph(tag_1000, "rmse")
            row = {
                "split": split,
                "horizon_min": int(h),
                "mae_pooled_300": mae_300,
                "mae_pooled_1000": mae_1000,
                "mae_pooled_delta": mae_1000 - mae_300,
                "mae_pooled_rel_pct": 100.0 * (mae_1000 - mae_300) / mae_300 if mae_300 else float("nan"),
                "rmse_pooled_300": rmse_300,
                "rmse_pooled_1000": rmse_1000,
                "rmse_pooled_delta": rmse_1000 - rmse_300,
                "mae_hypo_300": _pz(tag_300, "hypo"),
                "mae_hypo_1000": _pz(tag_1000, "hypo"),
                "mae_hypo_delta": _pz(tag_1000, "hypo") - _pz(tag_300, "hypo"),
                "mae_tir_300": _pz(tag_300, "tir"),
                "mae_tir_1000": _pz(tag_1000, "tir"),
                "mae_tir_delta": _pz(tag_1000, "tir") - _pz(tag_300, "tir"),
                "mae_hyper_300": _pz(tag_300, "hyper"),
                "mae_hyper_1000": _pz(tag_1000, "hyper"),
                "mae_hyper_delta": _pz(tag_1000, "hyper") - _pz(tag_300, "hyper"),
                "mae_pat_avg_300": _pa(tag_300, "mae"),
                "mae_pat_avg_1000": _pa(tag_1000, "mae"),
                "mae_pat_avg_delta": _pa(tag_1000, "mae") - _pa(tag_300, "mae"),
                "clarke_A_300": _clarke(tag_300, "A"),
                "clarke_A_1000": _clarke(tag_1000, "A"),
                "clarke_A_delta": _clarke(tag_1000, "A") - _clarke(tag_300, "A"),
                "clarke_D_300": _clarke(tag_300, "D"),
                "clarke_D_1000": _clarke(tag_1000, "D"),
                "clarke_D_delta": _clarke(tag_1000, "D") - _clarke(tag_300, "D"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decision rule: promote GBM-1000 to primary baseline if there is meaningful
# improvement on pooled MAE at the test split AND no regression in hypo zone.
#   - "meaningful improvement" := pooled MAE reduces by >= 2% relative at any
#     test horizon (a defensible threshold for an ablation; deeper retuning
#     could justify a smaller delta but 2% is the headline-level threshold).
#   - "no hypo regression" := per-zone hypo MAE on test does not increase by
#     more than +0.5 mg/dL at any horizon (allowing tiny noise).
# ---------------------------------------------------------------------------

PROMOTE_MAE_REL_PCT = -2.0      # -2 % or better (more negative is better)
HYPO_TOLERANCE_MGDL = 0.5       # hypo MAE may rise at most by this much


def apply_decision_rule(comp: pd.DataFrame) -> tuple[bool, str]:
    test = comp[comp["split"] == "test"].copy()
    best_rel = float(test["mae_pooled_rel_pct"].min())  # most negative wins
    worst_hypo_delta = float(test["mae_hypo_delta"].max())  # most positive is worst
    meets_mae = best_rel <= PROMOTE_MAE_REL_PCT
    meets_hypo = worst_hypo_delta <= HYPO_TOLERANCE_MGDL
    promote = bool(meets_mae and meets_hypo)
    reason = (
        f"best pooled MAE delta (test) = {best_rel:+.2f}% "
        f"({'pass' if meets_mae else 'fail'} vs <= {PROMOTE_MAE_REL_PCT:.1f}%); "
        f"worst hypo MAE delta (test) = {worst_hypo_delta:+.2f} mg/dL "
        f"({'pass' if meets_hypo else 'fail'} vs <= {HYPO_TOLERANCE_MGDL:+.1f})"
    )
    return promote, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        gbm_max_iter = 200
    else:
        gbm_max_iter = 1000

    tag_300 = "gbm_lr0.05_d8"           # matches the existing phase_b_*.csv rows
    tag_1000 = "gbm_lr0.05_d8_iter1000"

    bundles: list[dict] = []

    # ---------------- Load GBM-300 from joblib and re-evaluate ----------------
    ckpt_path = MODELS_DIR / "gbm_phase_b.joblib"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing GBM-300 checkpoint: {ckpt_path}. "
            "Run src/run_phase_b.py --gbm-only first."
        )
    print(f"\n[gbm-300] loading {ckpt_path}")
    ckpt = joblib.load(ckpt_path)
    gbm300 = _GBM300Reload(ckpt)
    print(f"[gbm-300] n_iters_used (original) = {gbm300.n_iters_used_}, "
          f"params = {ckpt.get('params')}")

    for split_name, sp in (("val", val), ("test", test)):
        yhat = gbm300.predict(sp["X_dyn"], sp["X_stat"])
        bundles.append(evaluate_model(sp["y"], yhat, sp["pid"], tag_300, split_name))

    # ---------------- Fit GBM-1000 with everything else identical -------------
    print(f"\n[gbm-1000] fitting max_iter={gbm_max_iter}, lr=0.05, max_depth=8, "
          f"early_stopping=True, n_iter_no_change=20, val_frac=0.1, seed={C.SEED}")
    t_gbm = time.time()
    gbm1000 = HistGBMBaseline(
        max_iter=gbm_max_iter,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=C.SEED,
    ).fit(
        train["X_dyn"], train["X_stat"], train["y"],
        feature_names_dynamic=data["feat_dyn"],
        feature_names_static=data["feat_stat"],
    )
    print(f"[gbm-1000] fit time = {time.time() - t_gbm:.1f}s  "
          f"n_iters_used per horizon = {gbm1000.n_iters_used_}")

    # Log iters_used and whether early stopping triggered
    pd.DataFrame({
        "horizon_min": list(C.HORIZON_MINUTES),
        "horizon_idx": list(range(len(C.HORIZON_MINUTES))),
        "n_iters_used": gbm1000.n_iters_used_,
        "max_iter_cap": gbm_max_iter,
        "early_stopped": [int(n < gbm_max_iter) for n in gbm1000.n_iters_used_],
    }).to_csv(TABLES_DIR / "phase_b_gbm1000_n_iters.csv", index=False)
    print("[save] phase_b_gbm1000_n_iters.csv")

    # Save model (do NOT overwrite GBM-300)
    out_model = MODELS_DIR / "gbm_phase_b_1000.joblib"
    joblib.dump(
        {
            "models": gbm1000.models_,
            "params": gbm1000.params,
            "n_iters_used": gbm1000.n_iters_used_,
            "feature_names": gbm1000.feature_names_,
        },
        out_model,
    )
    print(f"[save] {out_model.name}")

    for split_name, sp in (("val", val), ("test", test)):
        yhat = gbm1000.predict(sp["X_dyn"], sp["X_stat"])
        bundles.append(evaluate_model(sp["y"], yhat, sp["pid"], tag_1000, split_name))

    # ---------------- Save bundles and comparison ----------------
    print("\n[summary] writing aggregated tables")
    compact = save_bundles(bundles)
    comp = build_comparison_table(bundles, tag_300, tag_1000)
    comp.to_csv(TABLES_DIR / "phase_b_gbm_comparison.csv", index=False)
    print(f"[save] phase_b_gbm_comparison.csv              rows={len(comp)}")

    # ---------------- Print decision summary ----------------
    print("\n========== COMPACT SUMMARY (mg/dL) ==========")
    show = ["model", "split", "horizon_min", "mae", "rmse",
            "mae_pat_avg", "rmse_pat_avg",
            "clarke_pct_A", "clarke_pct_D"]
    show = [c for c in show if c in compact.columns]
    print(compact[show].to_string(index=False))

    print("\n========== TEST-SET COMPARISON (GBM-1000 minus GBM-300) ==========")
    show_comp = [
        "split", "horizon_min",
        "mae_pooled_300", "mae_pooled_1000", "mae_pooled_delta", "mae_pooled_rel_pct",
        "mae_hypo_300", "mae_hypo_1000", "mae_hypo_delta",
        "mae_pat_avg_300", "mae_pat_avg_1000", "mae_pat_avg_delta",
        "clarke_A_300", "clarke_A_1000", "clarke_A_delta",
    ]
    print(comp[comp["split"] == "test"][show_comp].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    promote, reason = apply_decision_rule(comp)
    verdict = "PROMOTE GBM-1000 to primary baseline" if promote else "KEEP GBM-300 as primary (near convergence)"
    print(f"\n[decision] {verdict}")
    print(f"[decision] rule: {reason}")

    print(f"\n[done] elapsed = {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true",
                    help="5k train / 2k val / 2k test, max_iter=200 for smoke-test")
    args = ap.parse_args()
    raise SystemExit(main(debug=args.debug))

"""SKILL.md §5.6 post-training comparison artefacts for Step 5 (Phase A + B + C).

Loads every model checkpoint available on disk (``outputs/models/``), runs
inference on val + test, and produces the four §5.6 deliverables:

1. **Master prediction table** — one row per ``(model, split, sample_idx,
   horizon_min)`` with ``participant_id``, ``y_true``, ``y_pred``, absolute
   error, squared error, and glycaemic zone. Saved as parquet
   (``outputs/tables/all_models_predictions.parquet``) because the table is
   ~2 M rows for the full model set.

2. **Scatter plot grid (y_pred vs y_true)** — one panel per
   ``(model, horizon)`` with the ``y = x`` identity line, drawn for the
   test split. Saved as ``outputs/figures/05_scatter_pred_vs_true.png``.

3. **Residual distribution by horizon × zone** — histograms of
   ``y_pred - y_true`` faceted by zone (hypo/tir/hyper) × horizon for each
   model. Saved as ``outputs/figures/05_residuals_by_zone.png``.

4. **Time-series overlay** — actual glucose vs. each model's 30 min
   forecast over a 24-hour window of the test split, for two
   representative patients (one long, one medium-short). Saved as
   ``outputs/figures/05_timeseries_overlay.png``.

Usage::

    python src/run_comparison_artefacts.py            # all available models
    python src/run_comparison_artefacts.py --models persistence,gru_c2_zwh30a

Notes:

* The script silently skips any model checkpoint that is missing. After
  Phase C.2 finishes, re-run to pick up the new GRU C.2 ``.pt`` files.
* LSTM/GRU C.1 ``.pt`` files are not present locally as of 2026-05-19
  (Colab Drive only). The script will note them as missing and proceed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from baselines import PersistenceModel, flatten_window  # noqa: E402
from evaluate import ZONE_LABELS, zone_of  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

PREDICTIONS_PARQUET = TABLES_DIR / "all_models_predictions.parquet"

# Catalogue of models the artefacts script knows how to load.
# ``kind`` selects the predict function; ``path`` is relative to MODELS_DIR.
MODEL_CATALOGUE: list[dict] = [
    {"name": "persistence",   "kind": "persistence",      "path": None},
    {"name": "ridge_a0.1",    "kind": "joblib_sklearn",   "path": "ridge_phase_a.joblib"},
    {"name": "rf_n300",       "kind": "joblib_sklearn",   "path": "rf_phase_b.joblib"},
    {"name": "gbm_n300",      "kind": "joblib_gbm_heads", "path": "gbm_phase_b.joblib"},
    {"name": "lstm_c1",       "kind": "pt_recurrent",     "path": "lstm_phase_c1.pt"},
    {"name": "gru_c1",        "kind": "pt_recurrent",     "path": "gru_phase_c1.pt"},
    {"name": "gru_c2_zw",     "kind": "pt_recurrent",     "path": "gru_c2_zw.pt"},
    {"name": "gru_c2_zwh30",  "kind": "pt_recurrent",     "path": "gru_c2_zwh30.pt"},
    {"name": "gru_c2_zwh30a", "kind": "pt_recurrent",     "path": "gru_c2_zwh30a.pt"},
    {"name": "step6_hybrid",  "kind": "pt_hybrid",        "path": "step6_hybrid.pt"},
]


# ---------------------------------------------------------------------------
# Data loading
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
    return out


def slice_split(data: dict, name: str) -> dict:
    mask = data["split"] == name
    return {
        "X_dyn": data["X_dyn"][mask],
        "X_stat": data["X_stat"][mask],
        "y": data["y"][mask],
        "pid": data["pid"][mask],
        "sample_idx": np.flatnonzero(mask).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Per-kind predict functions
# ---------------------------------------------------------------------------

def predict_persistence(sp: dict, feat_dyn: list[str]) -> np.ndarray:
    glucose_idx = feat_dyn.index("glucose")
    pers = PersistenceModel.from_scalers_json(
        PROJECT_ROOT / C.SCALERS_JSON, glucose_feature_index=glucose_idx,
    )
    return pers.predict(sp["X_dyn"], sp["pid"], n_horizons=len(C.HORIZON_MINUTES))


def predict_joblib_sklearn(model_path: Path, sp: dict) -> np.ndarray:
    import joblib
    bundle = joblib.load(model_path)
    model = bundle["model"]
    X = flatten_window(sp["X_dyn"], sp["X_stat"])
    return model.predict(X).astype(np.float32)


def predict_joblib_gbm_heads(model_path: Path, sp: dict) -> np.ndarray:
    import joblib
    bundle = joblib.load(model_path)
    heads = bundle["models"]
    X = flatten_window(sp["X_dyn"], sp["X_stat"])
    preds = np.stack([m.predict(X) for m in heads], axis=1)
    return preds.astype(np.float32)


def _predict_torch_module(model, sp: dict) -> np.ndarray:
    import torch
    model.eval()
    preds = []
    bs = 512
    with torch.no_grad():
        for i in range(0, sp["X_dyn"].shape[0], bs):
            xd = torch.from_numpy(sp["X_dyn"][i:i + bs])
            xs = torch.from_numpy(sp["X_stat"][i:i + bs])
            preds.append(model(xd, xs).numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def predict_pt_recurrent(model_path: Path, sp: dict, feat_dyn: list[str], feat_stat: list[str]) -> np.ndarray:
    import torch
    from models import RecurrentRegressor

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    n_dyn = cfg.get("n_dynamic", len(feat_dyn))
    n_stat = cfg.get("n_static", len(feat_stat))
    rnn_type = cfg.get("rnn_type", "gru")
    model = RecurrentRegressor(
        rnn_type=rnn_type,
        n_dynamic=n_dyn,
        n_static=n_stat,
        n_horizons=cfg.get("n_horizons", len(C.HORIZON_MINUTES)),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        static_embed_dim=cfg.get("static_embed_dim", 32),
        head_hidden_dim=cfg.get("head_hidden_dim", 64),
        bidirectional=cfg.get("bidirectional", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return _predict_torch_module(model, sp)


def predict_pt_hybrid(model_path: Path, sp: dict, feat_dyn: list[str], feat_stat: list[str]) -> np.ndarray:
    import torch
    from models import HybridCNNGRU

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    n_dyn = cfg.get("n_dynamic", len(feat_dyn))
    n_stat = cfg.get("n_static", len(feat_stat))
    model = HybridCNNGRU(
        n_dynamic=n_dyn,
        n_static=n_stat,
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
    model.load_state_dict(ckpt["model_state_dict"])
    return _predict_torch_module(model, sp)


# ---------------------------------------------------------------------------
# Master prediction table
# ---------------------------------------------------------------------------

def build_predictions_df(
    model_name: str, split_name: str, sp: dict, y_pred: np.ndarray,
) -> pd.DataFrame:
    """Tall ``(n_samples * n_horizons, 9)`` frame for one (model, split)."""
    horizons = C.HORIZON_MINUTES
    n = sp["y"].shape[0]
    pid = sp["pid"]
    sample_idx = sp["sample_idx"]
    frames = []
    for h_idx, h in enumerate(horizons):
        yt = sp["y"][:, h_idx]
        yp = y_pred[:, h_idx]
        err = yp - yt
        zones = zone_of(yt)
        df = pd.DataFrame({
            "model": model_name,
            "split": split_name,
            "sample_idx": sample_idx,
            "participant_id": pid,
            "horizon_min": int(h),
            "y_true": yt.astype(np.float32),
            "y_pred": yp.astype(np.float32),
            "abs_err": np.abs(err).astype(np.float32),
            "sq_err": (err ** 2).astype(np.float32),
            "zone": zones,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_predictions(
    data: dict,
    selected_models: list[dict],
) -> pd.DataFrame:
    splits = {name: slice_split(data, name) for name in ("val", "test")}
    feat_dyn = data["feat_dyn"]
    feat_stat = data["feat_stat"]

    all_frames: list[pd.DataFrame] = []
    for spec in selected_models:
        name = spec["name"]
        kind = spec["kind"]
        path = MODELS_DIR / spec["path"] if spec["path"] else None
        if path is not None and not path.exists():
            print(f"[skip] {name:18s} kind={kind:18s} (missing {path.name})")
            continue
        print(f"[predict] {name:18s} kind={kind:18s} path={getattr(path, 'name', '-')}")
        t0 = time.time()
        for split_name, sp in splits.items():
            if kind == "persistence":
                y_pred = predict_persistence(sp, feat_dyn)
            elif kind == "joblib_sklearn":
                y_pred = predict_joblib_sklearn(path, sp)
            elif kind == "joblib_gbm_heads":
                y_pred = predict_joblib_gbm_heads(path, sp)
            elif kind == "pt_recurrent":
                y_pred = predict_pt_recurrent(path, sp, feat_dyn, feat_stat)
            elif kind == "pt_hybrid":
                y_pred = predict_pt_hybrid(path, sp, feat_dyn, feat_stat)
            else:
                raise ValueError(f"unknown kind {kind!r}")
            df = build_predictions_df(name, split_name, sp, y_pred)
            all_frames.append(df)
        print(f"           {time.time() - t0:5.1f}s  ({sum(len(f) for f in all_frames[-2:])} rows added)")

    if not all_frames:
        raise RuntimeError("no model produced predictions — nothing to save")
    full = pd.concat(all_frames, ignore_index=True)
    return full


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scatter_grid(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    test = df[df["split"] == "test"]
    models = sorted(test["model"].unique())
    horizons = sorted(test["horizon_min"].unique())
    n_rows = len(models)
    n_cols = len(horizons)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    lim = (20, 460)
    for r, m in enumerate(models):
        for c, h in enumerate(horizons):
            ax = axes[r][c]
            sub = test[(test["model"] == m) & (test["horizon_min"] == h)]
            if len(sub) > 8000:
                sub = sub.sample(8000, random_state=C.SEED)
            ax.scatter(sub["y_true"], sub["y_pred"], s=2, alpha=0.15, color="#1f77b4")
            ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.6)
            for thr in (C.GLUCOSE_HYPO_THRESHOLD, C.GLUCOSE_HYPER_THRESHOLD):
                ax.axvline(thr, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
                ax.axhline(thr, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            if r == 0:
                ax.set_title(f"{h} min")
            if c == 0:
                ax.set_ylabel(f"{m}\ny_pred (mg/dL)")
            if r == n_rows - 1:
                ax.set_xlabel("y_true (mg/dL)")
    fig.suptitle("§5.6 — Test predictions vs. actual (y = x dashed; grey lines at 70 and 180 mg/dL)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.name}")


def plot_residuals_by_zone(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    test = df[df["split"] == "test"].copy()
    test["residual"] = test["y_pred"] - test["y_true"]
    models = sorted(test["model"].unique())
    horizons = sorted(test["horizon_min"].unique())
    zones = list(ZONE_LABELS)
    n_rows = len(horizons)
    n_cols = len(zones)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 2.4 * n_rows),
        sharex=False, sharey=False, squeeze=False,
    )
    palette = plt.get_cmap("tab10")
    for r, h in enumerate(horizons):
        for c, z in enumerate(zones):
            ax = axes[r][c]
            for i, m in enumerate(models):
                sub = test[
                    (test["model"] == m) & (test["horizon_min"] == h) & (test["zone"] == z)
                ]
                if sub.empty:
                    continue
                res = sub["residual"].to_numpy()
                xlim = np.percentile(res, [1, 99])
                ax.hist(
                    res, bins=60, range=tuple(xlim), histtype="step",
                    label=m, color=palette(i % 10), linewidth=1.1,
                )
            ax.axvline(0, color="k", linewidth=0.7, linestyle="--")
            if r == 0:
                ax.set_title(f"zone={z}")
            if c == 0:
                ax.set_ylabel(f"{h} min\ncount")
            if r == n_rows - 1:
                ax.set_xlabel("residual y_pred - y_true (mg/dL)")
    axes[0][-1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("§5.6 — Test residual distributions by horizon × glycaemic zone", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.name}")


def pick_representative_patients(df: pd.DataFrame) -> list[str]:
    """Pick (long, medium) patients by test-split sample count."""
    test = df[df["split"] == "test"]
    counts = test.groupby("participant_id")["sample_idx"].nunique().sort_values()
    if counts.empty:
        return []
    long_pid = counts.index[-1]  # longest in test
    median_pid = counts.index[len(counts) // 2]
    return [str(median_pid), str(long_pid)]


def plot_timeseries_overlay(df: pd.DataFrame, out_path: Path, window_steps: int = 288) -> None:
    """Show a ~24 h window of test data for two representative patients.

    Uses the 30 min horizon column for clarity (1 panel per patient).
    """
    import matplotlib.pyplot as plt

    pids = pick_representative_patients(df)
    if not pids:
        print("[skip] timeseries overlay (no test patients)")
        return
    test = df[(df["split"] == "test") & (df["horizon_min"] == 30)].copy()
    models = sorted(test["model"].unique())
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(pids), 1, figsize=(11, 3.0 * len(pids)), sharex=False, squeeze=False)
    for r, pid in enumerate(pids):
        sub = test[test["participant_id"] == pid].sort_values("sample_idx")
        if sub.empty:
            continue
        start_idx = sub["sample_idx"].iloc[0]
        window = sub[sub["sample_idx"] < start_idx + window_steps]
        if window.empty:
            continue
        ax = axes[r][0]
        x_steps = window.groupby("sample_idx")["y_true"].first().index.to_numpy()
        x_t = (x_steps - x_steps.min()) * C.SAMPLING_STEP_MIN
        y_t = window.groupby("sample_idx")["y_true"].first().to_numpy()
        ax.plot(x_t, y_t, color="black", linewidth=1.4, label="actual")
        for i, m in enumerate(models):
            yp = window[window["model"] == m].sort_values("sample_idx")["y_pred"].to_numpy()
            if yp.shape[0] != x_t.shape[0]:
                continue
            ax.plot(x_t, yp, color=palette(i % 10), linewidth=0.9, alpha=0.85, label=m)
        for thr in (C.GLUCOSE_HYPO_THRESHOLD, C.GLUCOSE_HYPER_THRESHOLD):
            ax.axhline(thr, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_title(f"{pid} — 30 min forecast vs actual (~24 h test window)")
        ax.set_ylabel("glucose (mg/dL)")
        if r == len(pids) - 1:
            ax.set_xlabel("minutes from window start")
        if r == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    fig.suptitle("§5.6 — Time-series overlay (30 min horizon)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(models_filter: tuple[str, ...] | None) -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if models_filter:
        selected = [m for m in MODEL_CATALOGUE if m["name"] in models_filter]
        missing = set(models_filter) - {m["name"] for m in selected}
        if missing:
            print(f"[warn] unknown model names ignored: {sorted(missing)}")
    else:
        selected = list(MODEL_CATALOGUE)
    print(f"[info] {len(selected)} model(s) selected; checking checkpoints under {MODELS_DIR}")

    data = load_data()
    t0 = time.time()
    full = run_predictions(data, selected)
    print(f"[info] predictions concatenated: rows={len(full):,}  elapsed={time.time() - t0:.1f}s")

    full.to_parquet(PREDICTIONS_PARQUET, index=False)
    print(f"[save] {PREDICTIONS_PARQUET.name}  ({PREDICTIONS_PARQUET.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity: mean MAE per (model, horizon) on test
    test = full[full["split"] == "test"]
    summary = (
        test.groupby(["model", "horizon_min"])["abs_err"].mean().reset_index(name="mae")
            .pivot(index="model", columns="horizon_min", values="mae")
    )
    summary.columns = [f"mae_{int(h)}m" for h in summary.columns]
    summary = summary.reset_index().sort_values("mae_30m")
    print("\n========== test pooled MAE (mg/dL) ==========")
    print(summary.to_string(index=False))
    summary.to_csv(TABLES_DIR / "all_models_test_mae_summary.csv", index=False)
    print(f"[save] all_models_test_mae_summary.csv  rows={len(summary)}")

    plot_scatter_grid(full, FIGURES_DIR / "05_scatter_pred_vs_true.png")
    plot_residuals_by_zone(full, FIGURES_DIR / "05_residuals_by_zone.png")
    plot_timeseries_overlay(full, FIGURES_DIR / "05_timeseries_overlay.png")

    print(f"\n[done] total elapsed = {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default=None,
        help="comma-separated subset of model names; default: all available",
    )
    args = ap.parse_args()
    filter_set = tuple(s.strip() for s in args.models.split(",")) if args.models else None
    raise SystemExit(main(models_filter=filter_set))

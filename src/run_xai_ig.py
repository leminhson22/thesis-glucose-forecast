"""Runner for §10 Explainability — Integrated Gradients on PersResid.

Produces:

  outputs/tables/xai_ig_global_heatmap_h{30,60,90}.csv
      Aggregated |IG| heat-map (24 timesteps × 17 dynamic features) over a
      large random sample of test windows. Used for the global-importance
      narrative in §10.

  outputs/tables/xai_ig_global_importance.csv
      Per-horizon feature ranking (summed |IG| across timesteps and samples).

  outputs/tables/xai_ig_temporal_focus.csv
      Per-horizon timestep-importance share (does the model attend to
      recent or remote lookback steps).

  outputs/tables/xai_ig_case_studies.parquet
      Per-sample attributions for ~30 curated case studies (10 worst hypo
      cases, 10 worst hyper cases, 10 representative TIR cases at 30 min).

  outputs/figures/10_ig_global_heatmap.png
      Three-panel heatmap (one per horizon), 24×17.

  outputs/figures/10_ig_feature_ranking.png
      Horizontal bar chart of top-10 dynamic feature importances at each horizon.

  outputs/figures/10_ig_temporal_focus.png
      Three-panel line plot showing |IG| share by lookback timestep per horizon.

Usage::

    python src/run_xai_ig.py
    python src/run_xai_ig.py --n-global 500 --n-cases 30 --ig-steps 50
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
from eval_step6_v2 import load_variant_model, predict_on_split  # noqa: E402
from explain import (  # noqa: E402
    global_feature_importance,
    integrated_gradients_dyn,
    temporal_feature_heatmap,
    temporal_focus,
)
from run_step6_v2 import attach_pid_index_to_static, load_pid_scaler_table  # noqa: E402

PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def stratified_sample(zones: np.ndarray, n_per_zone: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for z in ("hypo", "tir", "hyper"):
        idx = np.where(zones == z)[0]
        if len(idx) == 0:
            continue
        take = min(n_per_zone, len(idx))
        out.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(out))


def zone_for_each(y_true: np.ndarray) -> np.ndarray:
    z = np.full(len(y_true), "tir", dtype=object)
    z[y_true < 70] = "hypo"
    z[y_true > 180] = "hyper"
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-global", type=int, default=600,
                    help="Test-window sample size for the global heat-map (200 per zone).")
    ap.add_argument("--n-cases", type=int, default=30,
                    help="Number of curated case-study windows (10 worst hypo "
                         "+ 10 worst hyper + 10 representative TIR at 30m).")
    ap.add_argument("--ig-steps", type=int, default=50,
                    help="Riemann interpolation steps for IG.")
    ap.add_argument("--batch", type=int, default=8,
                    help="Sample-batch size during IG (memory-bound; reduce if OOM).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[xai] loading sequences from {C.SEQUENCES_NPZ}")
    splits = load_npz_splits(PROJECT_ROOT / C.SEQUENCES_NPZ)
    feat_dyn = splits["feat_dyn"]
    print(f"[xai] {len(feat_dyn)} dynamic features: {feat_dyn}")

    # PersResid needs the pid column appended to X_static
    _, _, pid_lookup = load_pid_scaler_table(splits)
    work = attach_pid_index_to_static(splits, pid_lookup)

    ckpt = MODELS_DIR / "step6_hybrid_v2_pers_resid.pt"
    print(f"[xai] loading checkpoint {ckpt.name}")
    model = load_variant_model(
        "pers_resid", ckpt,
        n_dynamic=len(feat_dyn),
        n_static_dataset=work["train"]["X_static"].shape[1],
        feat_dyn=feat_dyn, splits=splits,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Test split arrays
    # ------------------------------------------------------------------
    test = work["test"]
    X_dyn = test["X_dynamic"]
    X_stat = test["X_static"]
    y_true_all = test["y"]                       # (N, 3)
    pids = test["pids"]
    n_samples = X_dyn.shape[0]
    horizons = list(C.HORIZON_MINUTES)
    print(f"[xai] test set: {n_samples:,} windows, horizons={horizons}")

    # ------------------------------------------------------------------
    # GLOBAL HEAT-MAP — stratified random sample
    # ------------------------------------------------------------------
    n_per_zone = max(50, args.n_global // 3)
    zones_30m = zone_for_each(y_true_all[:, 0])
    idx_global = stratified_sample(zones_30m, n_per_zone, seed=args.seed)
    print(f"[xai] global sample: {len(idx_global)} windows "
          f"(per-zone target {n_per_zone}).")

    xd_g = torch.from_numpy(X_dyn[idx_global]).float()
    xs_g = torch.from_numpy(X_stat[idx_global]).float()

    global_attribs: dict[int, torch.Tensor] = {}
    for h_idx, h in enumerate(horizons):
        print(f"[xai] global IG, horizon={h}min ...")
        # Batched IG over chunks to keep peak memory low
        chunks = []
        for start in range(0, xd_g.shape[0], args.batch):
            end = min(xd_g.shape[0], start + args.batch)
            a = integrated_gradients_dyn(
                model,
                xd_g[start:end], xs_g[start:end],
                horizon_idx=h_idx, m=args.ig_steps,
            )
            chunks.append(a)
        global_attribs[int(h)] = torch.cat(chunks, dim=0)
        print(f"[xai] horizon={h}min attributions: {global_attribs[int(h)].shape}")

    # Save heat-maps and rankings per horizon
    importance_rows = []
    focus_rows = []
    for h in horizons:
        a = global_attribs[int(h)]
        # 2-D heat-map (timestep × feature)
        hm = temporal_feature_heatmap(a, feat_dyn, aggregate="abs_mean")
        hm.to_csv(TABLES_DIR / f"xai_ig_global_heatmap_h{h}.csv")
        # 1-D feature ranking
        imp = global_feature_importance(a, feat_dyn)
        imp["horizon_min"] = int(h)
        importance_rows.append(imp)
        # 1-D timestep focus
        foc = temporal_focus(a)
        foc = foc.reset_index().rename(columns={"timestep": "lookback_step"})
        foc["horizon_min"] = int(h)
        focus_rows.append(foc)

    pd.concat(importance_rows, ignore_index=True).to_csv(
        TABLES_DIR / "xai_ig_global_importance.csv", index=False
    )
    pd.concat(focus_rows, ignore_index=True).to_csv(
        TABLES_DIR / "xai_ig_temporal_focus.csv", index=False
    )
    print("[xai] wrote xai_ig_global_*.csv")

    # ------------------------------------------------------------------
    # CASE STUDIES — worst hypo / hyper, representative TIR at 30 min
    # ------------------------------------------------------------------
    # Run model on the full test set to get y_pred for ranking
    print("[xai] computing test predictions for case selection ...")
    y_pred = predict_on_split(model, test, batch_size=512)
    abs_err_30 = np.abs(y_pred[:, 0] - y_true_all[:, 0])
    zones = zones_30m
    n_each = args.n_cases // 3

    hypo_idx = np.where(zones == "hypo")[0]
    hypo_pick = hypo_idx[np.argsort(abs_err_30[hypo_idx])[-n_each:]]
    hyper_idx = np.where(zones == "hyper")[0]
    hyper_pick = hyper_idx[np.argsort(abs_err_30[hyper_idx])[-n_each:]]
    tir_idx = np.where(zones == "tir")[0]
    tir_pick = rng.choice(tir_idx, size=min(n_each, len(tir_idx)), replace=False)
    case_idx = np.sort(np.concatenate([hypo_pick, hyper_pick, tir_pick]))
    print(f"[xai] case studies: {len(hypo_pick)} hypo + {len(hyper_pick)} hyper + "
          f"{len(tir_pick)} TIR = {len(case_idx)} total")

    xd_c = torch.from_numpy(X_dyn[case_idx]).float()
    xs_c = torch.from_numpy(X_stat[case_idx]).float()

    # IG attributions per horizon for each case
    case_rows = []
    for h_idx, h in enumerate(horizons):
        chunks = []
        for start in range(0, xd_c.shape[0], args.batch):
            end = min(xd_c.shape[0], start + args.batch)
            a = integrated_gradients_dyn(
                model, xd_c[start:end], xs_c[start:end],
                horizon_idx=h_idx, m=args.ig_steps,
            ).numpy()
            chunks.append(a)
        a_h = np.concatenate(chunks, axis=0)        # (N_cases, T, F)

        # Encode as long-form for parquet
        for k, idx in enumerate(case_idx):
            for t in range(a_h.shape[1]):
                for f, fname in enumerate(feat_dyn):
                    case_rows.append({
                        "case_id": int(idx),
                        "participant_id": pids[idx],
                        "horizon_min": int(h),
                        "lookback_step": int(t),
                        "feature": fname,
                        "attribution": float(a_h[k, t, f]),
                        "y_true": float(y_true_all[idx, h_idx]),
                        "y_pred": float(y_pred[idx, h_idx]),
                        "abs_err": float(abs(y_pred[idx, h_idx] - y_true_all[idx, h_idx])),
                        "zone_30m": zones[idx],
                    })
    case_df = pd.DataFrame(case_rows)
    case_path = TABLES_DIR / "xai_ig_case_studies.parquet"
    case_df.to_parquet(case_path, index=False)
    print(f"[xai] wrote {case_path}  rows={len(case_df):,}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n[xai] === GLOBAL TOP-10 dynamic features per horizon ===")
    g = pd.read_csv(TABLES_DIR / "xai_ig_global_importance.csv")
    for h in horizons:
        sub = g[g["horizon_min"] == h].head(10)
        print(f"\nHorizon = {h} min")
        print(sub[["feature", "importance_pct"]].to_string(index=False))

    print("\n[xai] === Temporal focus (share of |IG| by lookback step) ===")
    f = pd.read_csv(TABLES_DIR / "xai_ig_temporal_focus.csv")
    # Show top-5 most-attended timesteps per horizon
    for h in horizons:
        sub = f[f["horizon_min"] == h].sort_values("share_pct", ascending=False).head(5)
        print(f"\nHorizon = {h} min, top-5 most-attended lookback steps (lookback step 23 = most recent):")
        print(sub[["lookback_step", "share_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()

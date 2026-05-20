"""Runner for §9.8 — Adaptive Conformal Inference on the PersResid model.

Applies per-zone Mondrian ACI (Gibbs & Candès 2021) to the held-out test
split, with calibration residuals taken from the val split (same setup as
§9.3). The point estimator is the same `step6_hybrid_v2_pers_resid`
checkpoint; ACI does *not* retrain anything — it adapts the conformal
mis-coverage rate `alpha_t` online per glycaemic zone to keep empirical
coverage close to nominal under non-exchangeability.

Outputs
-------
  outputs/tables/uq_aci_coverage.csv        — per-horizon × per-zone results
  outputs/tables/uq_aci_alpha_trajectory.parquet — full alpha_t per sample
  outputs/figures/09_uq_aci_coverage_vs_static.png — bar chart
  outputs/figures/09_uq_aci_alpha_trajectory.png  — line plot of alpha_t

Usage::

    python src/run_uq_aci.py
    python src/run_uq_aci.py --gamma 0.005 --alpha 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uncertainty as UQ  # noqa: E402

TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

HORIZONS = (30, 60, 90)
ZONE_ORDER = ("hypo", "tir", "hyper")
ZONE_COLOUR = {"hypo": "#d62728", "tir": "#2ca02c", "hyper": "#ff7f0e"}


def sort_residuals_by_zone(df_val: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for z in ZONE_ORDER:
        r = np.abs(df_val.loc[df_val["zone"] == z, "y_true"]
                   - df_val.loc[df_val["zone"] == z, "y_pred"]).to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        r.sort()
        out[z] = r
    return out


def run_one_horizon(
    df: pd.DataFrame,
    *,
    horizon: int,
    model_name: str,
    alpha: float,
    gamma: float,
) -> tuple[dict, pd.DataFrame]:
    val = df[(df["model"] == model_name) & (df["split"] == "val") & (df["horizon_min"] == horizon)]
    tst = df[(df["model"] == model_name) & (df["split"] == "test") & (df["horizon_min"] == horizon)].copy()
    if val.empty or tst.empty:
        raise ValueError(f"missing val/test rows for {model_name} @ {horizon}")

    cal_by_zone = sort_residuals_by_zone(val)
    # Sort test in chronological order per patient
    tst = tst.sort_values(["participant_id", "sample_idx"]).reset_index(drop=True)
    out = UQ.adaptive_conformal_inference(
        y_true=tst["y_true"].to_numpy(),
        y_pred=tst["y_pred"].to_numpy(),
        cal_residuals_sorted_by_zone=cal_by_zone,
        zones=tst["zone"].to_numpy(),
        alpha_target=alpha,
        gamma=gamma,
        return_trajectory=True,
    )

    tst["alpha_t"] = out["alpha_t"]
    tst["lower_aci"] = out["lower"]
    tst["upper_aci"] = out["upper"]
    tst["hit_aci"] = out["hit"]
    tst["horizon_min"] = horizon
    return out, tst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="step6_v2_pers_resid")
    ap.add_argument("--predictions",
                    default="outputs/tables/step6_v2_predictions.parquet")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.005)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / args.predictions)
    print(f"[aci] loaded predictions: {len(df):,} rows; model={args.model}")

    rows: list[dict] = []
    trajectories: list[pd.DataFrame] = []
    for h in HORIZONS:
        print(f"[aci] horizon = {h} min …")
        result, tst = run_one_horizon(
            df, horizon=h, model_name=args.model,
            alpha=args.alpha, gamma=args.gamma,
        )
        # Overall row
        rows.append({
            "model": args.model, "horizon_min": h, "zone": "all",
            "n": len(tst),
            "coverage_pct": 100 * result["empirical_coverage"],
            "mean_width": result["mean_width"],
            "final_alpha": float("nan"),
            "alpha_target": args.alpha, "gamma": args.gamma,
        })
        # Per-zone rows
        for z in ZONE_ORDER:
            rows.append({
                "model": args.model, "horizon_min": h, "zone": z,
                "n": result["n_by_zone"][z],
                "coverage_pct": 100 * result["empirical_coverage_by_zone"][z],
                "mean_width": result["mean_width_by_zone"][z],
                "final_alpha": result["final_alpha_by_zone"][z],
                "alpha_target": args.alpha, "gamma": args.gamma,
            })
        trajectories.append(tst[[
            "participant_id", "sample_idx", "horizon_min",
            "y_true", "y_pred", "zone",
            "alpha_t", "lower_aci", "upper_aci", "hit_aci",
        ]])

    cov_df = pd.DataFrame(rows).round({"coverage_pct": 2, "mean_width": 2, "final_alpha": 4})
    cov_path = TBL / "uq_aci_coverage.csv"
    cov_df.to_csv(cov_path, index=False)
    print(f"[aci] wrote {cov_path}")

    traj = pd.concat(trajectories, ignore_index=True)
    traj_path = TBL / "uq_aci_alpha_trajectory.parquet"
    traj.to_parquet(traj_path, index=False)
    print(f"[aci] wrote {traj_path} ({len(traj):,} rows)")

    # ----------------------------- figures -----------------------------
    # Coverage bar chart (Mondrian-Split vs ACI per zone × horizon)
    mond_cov = pd.read_csv(TBL / "uq_conformal_coverage.csv")
    mond_cov = mond_cov[
        (mond_cov["model"] == args.model)
        & (mond_cov["method"] == "mondrian")
        & (np.isclose(mond_cov["alpha"], args.alpha))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    bar_w = 0.35
    for ax, h in zip(axes, HORIZONS):
        x = np.arange(len(ZONE_ORDER) + 1)
        zones = ["all"] + list(ZONE_ORDER)
        mond_vals = []
        aci_vals = []
        for z in zones:
            m = mond_cov[(mond_cov["horizon_min"] == h) & (mond_cov["zone"] == z)]
            a = cov_df[(cov_df["horizon_min"] == h) & (cov_df["zone"] == z)]
            mond_vals.append(m["coverage_pct"].iloc[0] if len(m) else np.nan)
            aci_vals.append(a["coverage_pct"].iloc[0] if len(a) else np.nan)
        ax.bar(x - bar_w / 2, mond_vals, bar_w, color="#4c72b0",
               edgecolor="black", label="Mondrian-Split CP")
        ax.bar(x + bar_w / 2, aci_vals, bar_w, color="#55a868",
               edgecolor="black", label="Mondrian-ACI")
        ax.axhline(100 * (1 - args.alpha), color="black", linestyle="--",
                   linewidth=1.2, label=f"Nominal {int(100 * (1 - args.alpha))} %")
        ax.axhspan(100 * (1 - args.alpha) - 1, 100 * (1 - args.alpha) + 1,
                   color="black", alpha=0.07)
        ax.set_xticks(x)
        ax.set_xticklabels(zones)
        ax.set_title(f"Horizon = {h} min")
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_ylim(80, 100)
        ax.grid(axis="y", alpha=0.25)
        if h == HORIZONS[0]:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Figure 9.4 — Mondrian-Split CP vs Mondrian-ACI empirical coverage on the test split "
        f"(α_target = {args.alpha}, γ = {args.gamma})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    cov_fig_path = FIG / "09_uq_aci_coverage_vs_static.png"
    fig.savefig(cov_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aci] wrote {cov_fig_path}")

    # alpha_t trajectory per zone (using horizon 30 as representative)
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    h30 = traj[traj["horizon_min"] == 30].sort_values(
        ["participant_id", "sample_idx"]
    ).reset_index(drop=True)
    sample_step = np.arange(len(h30))
    for ax, z in zip(axes2, ZONE_ORDER):
        mask = h30["zone"] == z
        if mask.any():
            # plot only where zone matches; gaps are filled
            ax.plot(sample_step[mask], h30.loc[mask, "alpha_t"], "-",
                    color=ZONE_COLOUR[z], linewidth=0.7, alpha=0.7)
        ax.axhline(args.alpha, color="black", linestyle="--", linewidth=1.0,
                   label=f"α_target = {args.alpha}")
        ax.set_ylabel(f"α_t  ({z})")
        ax.grid(alpha=0.25)
        if z == ZONE_ORDER[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes2[-1].set_xlabel("Test-set sample index (chronological per-patient)")
    fig2.suptitle(
        "Figure 9.5 — Per-zone α_t trajectory under Mondrian-ACI, horizon = 30 min",
        fontsize=11, y=0.995,
    )
    fig2.tight_layout()
    traj_fig_path = FIG / "09_uq_aci_alpha_trajectory.png"
    fig2.savefig(traj_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[aci] wrote {traj_fig_path}")

    # Console summary
    print("\n[aci] empirical coverage (test split):")
    show = cov_df[cov_df["zone"] != "all"][[
        "horizon_min", "zone", "n", "coverage_pct", "mean_width", "final_alpha"
    ]]
    print(show.to_string(index=False))
    print("\n[aci] vs Mondrian-Split per-zone:")
    for h in HORIZONS:
        for z in ZONE_ORDER:
            m = mond_cov[(mond_cov["horizon_min"] == h) & (mond_cov["zone"] == z)]
            a = cov_df[(cov_df["horizon_min"] == h) & (cov_df["zone"] == z)]
            if len(m) and len(a):
                print(f"  {h:>2}min  {z:<5}  Mondrian {m['coverage_pct'].iloc[0]:.2f}%  "
                      f"->  ACI {a['coverage_pct'].iloc[0]:.2f}%  "
                      f"(delta {a['coverage_pct'].iloc[0] - m['coverage_pct'].iloc[0]:+.2f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

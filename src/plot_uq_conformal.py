"""Figures for §9 Uncertainty Quantification — Conformal Prediction.

Generates three figures from the artefacts produced by
``src/run_uq_conformal.py``:

  Fig 9.1  outputs/figures/09_uq_coverage_calibration.png
           Empirical vs nominal coverage, per horizon × method × zone.
           Includes the ±1 percentage-point coverage tolerance band.

  Fig 9.2  outputs/figures/09_uq_interval_width_by_zone.png
           Mondrian interval half-width (q in mg/dL) per zone × horizon × alpha.

  Fig 9.3  outputs/figures/09_uq_intervals_timeseries.png
           Two-panel time-series overlay of the predicted 90 % PI band
           (Mondrian) with the actual glucose trajectory, for one
           short-duration patient (HUPA0014P) and one long-duration patient
           (HUPA0027P), at the 30-minute horizon.

All figures are saved at 150 DPI and use the project's mg/dL convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = ROOT / "outputs" / "figures"
TBL_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ZONE_COLOURS = {"hypo": "#d62728", "tir": "#2ca02c", "hyper": "#ff7f0e", "all": "#1f77b4"}
ZONE_ORDER = ("hypo", "tir", "hyper")
ALPHA_COVERAGE = {0.10: 90.0, 0.20: 80.0}


def fig_coverage_calibration(cov: pd.DataFrame) -> Path:
    """Bar chart of empirical coverage vs nominal target, per method/zone/horizon/alpha."""
    horizons = (30, 60, 90)
    methods = ("split", "mondrian")
    alphas = sorted(cov["alpha"].unique())

    fig, axes = plt.subplots(
        len(alphas), len(horizons),
        figsize=(13, 3.6 * len(alphas)),
        sharey=True,
    )
    if len(alphas) == 1:
        axes = np.array([axes])

    for i, alpha in enumerate(alphas):
        nominal = ALPHA_COVERAGE[alpha]
        for j, h in enumerate(horizons):
            ax = axes[i, j]
            # Group bars by zone, two bars per zone (split, mondrian)
            zones_present = ["all"] + list(ZONE_ORDER)
            x = np.arange(len(zones_present))
            bar_w = 0.35

            split_vals = []
            mond_vals = []
            for z in zones_present:
                rs = cov[
                    (cov["alpha"] == alpha)
                    & (cov["horizon_min"] == h)
                    & (cov["method"] == "split")
                    & (cov["zone"] == z)
                ]
                rm = cov[
                    (cov["alpha"] == alpha)
                    & (cov["horizon_min"] == h)
                    & (cov["method"] == "mondrian")
                    & (cov["zone"] == z)
                ]
                split_vals.append(rs["coverage_pct"].iloc[0] if len(rs) else np.nan)
                mond_vals.append(rm["coverage_pct"].iloc[0] if len(rm) else np.nan)

            ax.bar(x - bar_w / 2, split_vals, bar_w,
                   label="Split CP", color="#4c72b0", edgecolor="black", linewidth=0.5)
            ax.bar(x + bar_w / 2, mond_vals, bar_w,
                   label="Mondrian CP", color="#dd8452", edgecolor="black", linewidth=0.5)

            # Nominal target line + tolerance band
            ax.axhline(nominal, color="black", linestyle="--", linewidth=1.2,
                       label=f"Nominal {int(nominal)} %")
            ax.axhspan(nominal - 1, nominal + 1, color="black", alpha=0.07,
                       label="±1 pp tolerance")

            ax.set_xticks(x)
            ax.set_xticklabels(["overall"] + [z for z in ZONE_ORDER])
            ax.set_title(f"Horizon = {h} min, α = {alpha:.2f}  (nominal {int(nominal)} % PI)")
            if j == 0:
                ax.set_ylabel("Empirical coverage (%)")
            ax.set_ylim(70, 100)
            ax.grid(axis="y", alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Figure 9.1 — Conformal Prediction empirical coverage on the test split "
        "(model = CNN-GRU-Attention + Persistence-Residual)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "09_uq_coverage_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_interval_width_by_zone(qt: pd.DataFrame) -> Path:
    """Half-width q (mg/dL) per zone × horizon × alpha, Mondrian only."""
    mond = qt[qt["method"] == "mondrian"].copy()
    horizons = (30, 60, 90)
    alphas = sorted(mond["alpha"].unique())

    fig, axes = plt.subplots(1, len(alphas), figsize=(11, 4.2), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    bar_w = 0.25
    for ax, alpha in zip(axes, alphas):
        x = np.arange(len(horizons))
        for k, z in enumerate(ZONE_ORDER):
            vals = [
                mond[(mond["alpha"] == alpha) & (mond["horizon_min"] == h) & (mond["zone"] == z)][
                    "q_mgdl"
                ].iloc[0]
                for h in horizons
            ]
            ax.bar(x + (k - 1) * bar_w, vals, bar_w,
                   label=z, color=ZONE_COLOURS[z], edgecolor="black", linewidth=0.5)
            for xx, vv in zip(x + (k - 1) * bar_w, vals):
                ax.text(xx, vv + 0.8, f"{vv:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h} min" for h in horizons])
        ax.set_title(f"α = {alpha:.2f}  (nominal {int(ALPHA_COVERAGE[alpha])} % PI)")
        ax.set_ylabel("Half-width q (mg/dL)")
        ax.legend(title="Zone (of last reference glucose)", fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Figure 9.2 — Mondrian Conformal Prediction half-widths by glycaemic zone\n"
        "(wider band = larger predictive uncertainty in that zone)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    out = FIG_DIR / "09_uq_interval_width_by_zone.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_intervals_timeseries(df: pd.DataFrame) -> Path:
    """Time-series overlay of the 90 % Mondrian PI for two representative patients."""
    # The intervals parquet has columns: lower_split_a10, upper_split_a10,
    # lower_mondrian_a10, upper_mondrian_a10, etc.
    patients = [("HUPA0014P", "Short-duration patient (HUPA0014P)"),
                ("HUPA0027P", "Long-duration patient (HUPA0027P)")]

    fig, axes = plt.subplots(len(patients), 1, figsize=(11, 7), sharex=False)

    for ax, (pid, title) in zip(axes, patients):
        sub = df[
            (df["participant_id"] == pid)
            & (df["horizon_min"] == 30)
        ].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No test data for {pid}", ha="center", va="center")
            ax.set_axis_off()
            continue
        sub = sub.sort_values("sample_idx").reset_index(drop=True)
        # Plot the first 24 hours of the patient's test window (~288 5-min ticks)
        window = sub.iloc[: min(288, len(sub))]

        t = np.arange(len(window)) * 5  # minutes from start of plotted window
        ax.plot(t, window["y_true"], color="black", linewidth=1.2, label="Actual glucose")
        ax.plot(t, window["y_pred"], color="#1f77b4", linewidth=1.0,
                alpha=0.9, label="PersResid forecast (+30 min)")
        ax.fill_between(
            t, window["lower_mondrian_a10"], window["upper_mondrian_a10"],
            color="#1f77b4", alpha=0.18, label="90 % Mondrian PI",
        )
        # Glycaemic threshold lines
        ax.axhline(70, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(180, color="#ff7f0e", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(0, 64, "70 mg/dL", color="#d62728", fontsize=7)
        ax.text(0, 184, "180 mg/dL", color="#ff7f0e", fontsize=7)

        ax.set_title(title)
        ax.set_xlabel("Time within plotted window (minutes)")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Figure 9.3 — Predicted 90 % Mondrian prediction interval (30-min horizon, test split)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "09_uq_intervals_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    cov = pd.read_csv(TBL_DIR / "uq_conformal_coverage.csv")
    qt = pd.read_csv(TBL_DIR / "uq_conformal_quantiles.csv")
    intervals = pd.read_parquet(TBL_DIR / "uq_conformal_intervals.parquet")

    p1 = fig_coverage_calibration(cov)
    p2 = fig_interval_width_by_zone(qt)
    p3 = fig_intervals_timeseries(intervals)

    print(f"[uq-plot] wrote {p1}")
    print(f"[uq-plot] wrote {p2}")
    print(f"[uq-plot] wrote {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figures for §10 Explainability — Integrated Gradients.

Generates three figures from artefacts produced by ``src/run_xai_ig.py``:

  Fig 10.1  outputs/figures/10_ig_global_heatmap.png
            Three-panel heat-map (one per horizon) of mean |IG| at every
            (lookback step, dynamic feature) cell. Row = feature, column =
            lookback step (oldest → most recent).

  Fig 10.2  outputs/figures/10_ig_feature_ranking.png
            Horizontal bar chart of top-10 dynamic-feature importances at
            each horizon (% of total |IG|).

  Fig 10.3  outputs/figures/10_ig_temporal_focus.png
            Line plot showing |IG| share by lookback step per horizon.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 90]


def fig_global_heatmap() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        hm = pd.read_csv(TBL / f"xai_ig_global_heatmap_h{h}.csv", index_col=0)
        # Sort rows by global importance (descending) for readability
        order = hm.sum(axis=0).sort_values(ascending=False).index.tolist()
        hm = hm[order]
        # hm columns = features, index = timesteps 0..23
        # We want features on Y axis, timesteps on X axis -> transpose
        mat = hm.T.values
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"t-{(23 - x) * 5}'" for x in range(0, 24, 4)], fontsize=8)
        ax.set_xlabel("Lookback step  (t-115min ← left, t-0 ← right)")
        ax.set_title(f"Horizon = {h} min")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Figure 10.1 — Global mean |IG| heat-map per horizon  "
        "(model = CNN-GRU-Attention + Persistence-Residual; "
        "rows sorted by global importance per panel)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = FIG / "10_ig_global_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_feature_ranking() -> Path:
    g = pd.read_csv(TBL / "xai_ig_global_importance.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        sub = g[g["horizon_min"] == h].head(10).iloc[::-1]
        bars = ax.barh(sub["feature"], sub["importance_pct"], color="#1f77b4",
                       edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, sub["importance_pct"]):
            ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                    va="center", fontsize=8)
        ax.set_title(f"Horizon = {h} min")
        ax.set_xlabel("Share of |IG| (%)")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Figure 10.2 — Top-10 dynamic-feature importance by Integrated Gradients per horizon",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = FIG / "10_ig_feature_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_temporal_focus() -> Path:
    f = pd.read_csv(TBL / "xai_ig_temporal_focus.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colours = {30: "#1f77b4", 60: "#ff7f0e", 90: "#d62728"}
    for h in HORIZONS:
        sub = f[f["horizon_min"] == h].sort_values("lookback_step")
        ax.plot(sub["lookback_step"], sub["share_pct"], "-o",
                color=colours[h], label=f"{h} min", linewidth=1.4, markersize=4)
    ax.set_xlabel("Lookback step (0 = 120 min ago, 23 = most recent)")
    ax.set_ylabel("Share of |IG| (%)")
    ax.set_title(
        "Figure 10.3 — Temporal focus of Integrated Gradients per horizon "
        "(higher share = the model relies more on that lookback step)"
    )
    ax.legend(title="Horizon", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xticks(range(0, 24, 2))
    out = FIG / "10_ig_temporal_focus.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p1 = fig_global_heatmap()
    p2 = fig_feature_ranking()
    p3 = fig_temporal_focus()
    print(f"[xai-plot] wrote {p1}")
    print(f"[xai-plot] wrote {p2}")
    print(f"[xai-plot] wrote {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

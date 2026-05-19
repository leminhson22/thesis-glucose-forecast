"""Feature selection analysis for HUPA-UCM (SKILL.md §4.5).

Computes three orthogonal feature-importance signals on the TRAIN portion
only (no leakage into val / test):

1. Spearman rank correlation between each feature and ``target_60m`` —
   measures monotonic association (catches non-linear monotonic patterns
   that Pearson misses).
2. Mutual information regression (sklearn) — captures non-linear,
   non-monotonic dependence; better for binary / categorical features.
3. Permutation feature importance from a Random Forest baseline — model-
   based, captures cross-feature interaction.

Two passes are run:

* **Per-timestep dynamic features.** Each dynamic feature is evaluated at the
  anchor timestep ``t`` (end of the lookback window). This is the "current
  value" of the feature at the moment of prediction, which is the most
  informative single instance of the lookback.
* **Static features.** Each static feature is evaluated on the same per-anchor
  sample, broadcast to the sequence-level granularity.

Output artefacts:
    outputs/tables/hupa_feature_selection.csv  — one row per feature
    outputs/figures/04_feature_selection_dynamic.png — top-20 dynamic bar chart
    outputs/figures/04_feature_selection_static.png  — static features ranking

Run from project root:
    python src/feature_selection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = PROJECT_ROOT / C.SEQUENCES_NPZ
STATIC_CSV = PROJECT_ROOT / C.STATIC_FEATURES_CSV
OUT_CSV = PROJECT_ROOT / "outputs" / "tables" / "hupa_feature_selection.csv"
OUT_FIG_DYN = PROJECT_ROOT / "outputs" / "figures" / "04_feature_selection_dynamic.png"
OUT_FIG_STAT = PROJECT_ROOT / "outputs" / "figures" / "04_feature_selection_static.png"

# Sub-sample to bound runtime. RF + permutation importance on 60K rows × 60
# features takes ~5–10 minutes; with a 20K sub-sample it runs in ~1 minute.
SAMPLE_SIZE = 20000
RF_N_ESTIMATORS = 80
RF_MAX_DEPTH = 12
N_PERMUTATION_REPEATS = 5
TARGET_HORIZON_MIN = 60   # use target_60m as the primary selection target
RANDOM_STATE = C.SEED


def load_anchor_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a per-anchor DataFrame whose rows are the train sequences.

    Each dynamic feature contributes one column: its value at the anchor
    timestep ``t`` (the last row of the lookback). Static features are
    appended as-is. The target ``target_60m`` is taken from the bundle's
    ``y`` array at the column corresponding to ``TARGET_HORIZON_MIN``.
    """
    print(f"Loading {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_dynamic = data["X_dynamic"]
    X_static = data["X_static"]
    y = data["y"]
    split = data["split"]
    feature_names_dynamic = list(data["feature_names_dynamic"])
    feature_names_static = list(data["feature_names_static"])
    horizon_minutes = list(data["horizon_minutes"])

    horizon_col = horizon_minutes.index(TARGET_HORIZON_MIN)

    train_mask = split == "train"
    print(f"  train rows in bundle: {train_mask.sum():,}")

    # Anchor value of dynamic features (last timestep of the lookback)
    dyn_anchor = X_dynamic[train_mask, -1, :]   # (N_train, n_dyn)
    stat = X_static[train_mask, :]              # (N_train, n_stat)
    target = y[train_mask, horizon_col]         # (N_train,)

    rng = np.random.default_rng(RANDOM_STATE)
    n = dyn_anchor.shape[0]
    if n > SAMPLE_SIZE:
        idx = rng.choice(n, size=SAMPLE_SIZE, replace=False)
        dyn_anchor = dyn_anchor[idx]
        stat = stat[idx]
        target = target[idx]
        print(f"  Sub-sampled to {SAMPLE_SIZE} rows for selection analysis")

    df = pd.DataFrame(dyn_anchor, columns=[f"dyn::{c}" for c in feature_names_dynamic])
    for j, c in enumerate(feature_names_static):
        df[f"stat::{c}"] = stat[:, j]
    df["target_60m"] = target
    return df, feature_names_dynamic, feature_names_static


def compute_metrics(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_60m"
) -> pd.DataFrame:
    """Run Spearman + mutual information for every feature."""
    print(f"  Computing Spearman + MI on {len(feature_cols)} features...")
    rows = []
    target = df[target_col].to_numpy(dtype=np.float64)

    # Mutual information in one call is fastest (sklearn handles vectorisation)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    mi = mutual_info_regression(X, target, random_state=RANDOM_STATE)

    for col, mi_val in zip(feature_cols, mi):
        x = df[col].to_numpy(dtype=np.float64)
        std = x.std()
        if std < 1e-12:
            rho, p = 0.0, 1.0
        else:
            rho, p = spearmanr(x, target)
        rows.append(
            {
                "feature": col,
                "spearman_r": rho,
                "spearman_abs": abs(rho),
                "mutual_information": float(mi_val),
            }
        )
    return pd.DataFrame(rows)


def compute_permutation(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_60m"
) -> pd.Series:
    """Train a small Random Forest then run permutation importance."""
    print(
        f"  Training RandomForestRegressor (n_estimators={RF_N_ESTIMATORS}, "
        f"max_depth={RF_MAX_DEPTH}) on {len(df):,} rows..."
    )
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    print(f"  Train R² = {rf.score(X, y):.3f}")

    print(f"  Computing permutation importance ({N_PERMUTATION_REPEATS} repeats)...")
    pi = permutation_importance(
        rf,
        X,
        y,
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.Series(pi.importances_mean, index=feature_cols, name="permutation_importance")


def add_composite_rank(table: pd.DataFrame) -> pd.DataFrame:
    """Combine the three signals into a single robust ranking.

    Each metric is converted to a rank (1 = strongest), then averaged. The
    composite rank is the final ordering used by §6 to decide what to keep.
    """
    out = table.copy()
    out["rank_spearman"] = out["spearman_abs"].rank(ascending=False, method="min")
    out["rank_mi"] = out["mutual_information"].rank(ascending=False, method="min")
    out["rank_perm"] = out["permutation_importance"].rank(ascending=False, method="min")
    out["rank_composite"] = (
        out["rank_spearman"] + out["rank_mi"] + out["rank_perm"]
    ) / 3.0
    return out.sort_values("rank_composite").reset_index(drop=True)


def plot_dynamic(table: pd.DataFrame, out_path: Path) -> None:
    sub = table[table["feature"].str.startswith("dyn::")].copy()
    sub["feature_short"] = sub["feature"].str.replace("dyn::", "", regex=False)
    sub = sub.head(20)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    metrics = [
        ("spearman_abs", "|Spearman r|", "tab:blue"),
        ("mutual_information", "Mutual information", "tab:green"),
        ("permutation_importance", "RF permutation importance", "tab:red"),
    ]
    for ax, (col, label, color) in zip(axes, metrics):
        s = sub.sort_values(col, ascending=True)
        ax.barh(s["feature_short"], s[col], color=color, alpha=0.85)
        ax.set_xlabel(label, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "Dynamic features — top 20 by composite rank "
        f"(target = glucose at t+{TARGET_HORIZON_MIN} min)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved {out_path}")


def plot_static(table: pd.DataFrame, out_path: Path) -> None:
    sub = table[table["feature"].str.startswith("stat::")].copy()
    sub["feature_short"] = sub["feature"].str.replace("stat::", "", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    metrics = [
        ("spearman_abs", "|Spearman r|", "tab:blue"),
        ("mutual_information", "Mutual information", "tab:green"),
        ("permutation_importance", "RF permutation importance", "tab:red"),
    ]
    for ax, (col, label, color) in zip(axes, metrics):
        s = sub.sort_values(col, ascending=True)
        ax.barh(s["feature_short"], s[col], color=color, alpha=0.85)
        ax.set_xlabel(label, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "Static features — full ranking "
        f"(target = glucose at t+{TARGET_HORIZON_MIN} min)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved {out_path}")


def main() -> None:
    df, fn_dyn, fn_stat = load_anchor_table()
    feature_cols = [c for c in df.columns if c != "target_60m"]

    metrics_table = compute_metrics(df, feature_cols)
    perm_series = compute_permutation(df, feature_cols)

    table = metrics_table.merge(
        perm_series.rename("permutation_importance").to_frame(),
        left_on="feature",
        right_index=True,
    )
    # If the merge produced duplicate columns (because Spearman frame already
    # has the same name), keep the perm column from compute_permutation.
    if "permutation_importance_y" in table.columns:
        table["permutation_importance"] = table["permutation_importance_y"]
        table = table.drop(
            columns=[c for c in table.columns if c.endswith("_y") or c.endswith("_x")
                     and c != "permutation_importance"]
        )
    table = add_composite_rank(table)

    table["kind"] = np.where(table["feature"].str.startswith("dyn::"), "dynamic", "static")
    table["feature_short"] = table["feature"].str.replace(r"^(dyn|stat)::", "", regex=True)

    cols_order = [
        "kind",
        "feature_short",
        "spearman_r",
        "spearman_abs",
        "mutual_information",
        "permutation_importance",
        "rank_spearman",
        "rank_mi",
        "rank_perm",
        "rank_composite",
    ]
    table_out = table[cols_order]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table_out.to_csv(OUT_CSV, index=False)
    print(f"  Saved {OUT_CSV}")

    plot_dynamic(table, OUT_FIG_DYN)
    plot_static(table, OUT_FIG_STAT)

    # Console summary
    print("\nTop 10 features overall (composite rank):")
    print(
        table_out.head(10).to_string(
            index=False, formatters={"spearman_r": "{:+.3f}".format,
                                     "spearman_abs": "{:.3f}".format,
                                     "mutual_information": "{:.3f}".format,
                                     "permutation_importance": "{:.4f}".format,
                                     "rank_composite": "{:.1f}".format}
        )
    )
    print("\nBottom 10 features (candidates to drop):")
    print(
        table_out.tail(10).to_string(
            index=False, formatters={"spearman_r": "{:+.3f}".format,
                                     "spearman_abs": "{:.3f}".format,
                                     "mutual_information": "{:.3f}".format,
                                     "permutation_importance": "{:.4f}".format,
                                     "rank_composite": "{:.1f}".format}
        )
    )


if __name__ == "__main__":
    main()

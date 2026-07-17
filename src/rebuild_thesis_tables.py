"""Rebuild Chapter 4 thesis result tables from tracked evaluation CSVs.

This script is the public, GitHub-friendly entrypoint for reproducing the
headline tables used in the thesis demo. It does not retrain models; it reads
the six-model evaluation summaries under ``outputs/tables`` and rewrites:

* ``outputs/tables/thesis/table_4_1_overall_mae_rmse_6models.{csv,md}``
* ``outputs/tables/thesis/table_4_2_zone_mae_6models.{csv,md}``
* ``outputs/tables/thesis/table_4_3_zone_rmse_6models.{csv,md}``

Run from the project root:

    python src/rebuild_thesis_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
THESIS = TABLES / "thesis"

MODEL_ORDER = [
    "persistence",
    "ridge_a0.1",
    "rf_n300",
    "lstm_phase_c1",
    "gru_phase_c1",
    "step6_hybrid_v2_pers_resid",
]

MODEL_LABELS = {
    "persistence": "Persistence",
    "ridge_a0.1": "Ridge Regression",
    "rf_n300": "Random Forest",
    "lstm_phase_c1": "LSTM",
    "gru_phase_c1": "GRU",
    "step6_hybrid_v2_pers_resid": "Hybrid CNN-GRU (proposed)",
}

SUMMARY_FILES = [
    "phase_a_summary.csv",
    "phase_b_summary.csv",
    "phase_c1_summary.csv",
    "step6_v2_pers_resid_summary.csv",
]

ZONE_FILES = [
    "phase_a_per_zone.csv",
    "phase_b_per_zone.csv",
    "phase_c1_per_zone.csv",
    "step6_v2_pers_resid_per_zone.csv",
]

HORIZONS = [30, 60, 90]
ZONE_ORDER = ["hypo", "tir", "hyper"]
ZONE_LABELS = {"hypo": "Hypo", "tir": "TIR", "hyper": "Hyper"}


def read_tables(names: list[str]) -> pd.DataFrame:
    frames = []
    missing = []
    for name in names:
        path = TABLES / name
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        frames.append(pd.read_csv(path))
    if missing:
        raise FileNotFoundError("Missing required input table(s): " + ", ".join(missing))
    return pd.concat(frames, ignore_index=True)


def write_table(df: pd.DataFrame, stem: str) -> None:
    THESIS.mkdir(parents=True, exist_ok=True)
    csv_path = THESIS / f"{stem}.csv"
    md_path = THESIS / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f"[write] {csv_path.relative_to(ROOT)}")
    print(f"[write] {md_path.relative_to(ROOT)}")


def rebuild_table_4_1() -> pd.DataFrame:
    summary = read_tables(SUMMARY_FILES)
    test = summary[(summary["split"] == "test") & (summary["model"].isin(MODEL_ORDER))]

    rows = []
    mae_rows = []
    for model in MODEL_ORDER:
        row = {"Model": MODEL_LABELS[model]}
        mae_row = {"model": model}
        sub = test[test["model"] == model]
        for h in HORIZONS:
            one = sub[sub["horizon_min"] == h]
            if one.empty:
                raise ValueError(f"Missing test summary for {model} @ {h} min")
            mae_value = float(one["mae"].iloc[0])
            row[f"{h}m MAE"] = round(mae_value, 2)
            row[f"{h}m RMSE"] = round(float(one["rmse"].iloc[0]), 2)
            mae_row[f"mae_{h}m"] = mae_value
        rows.append(row)
        mae_rows.append(mae_row)
    out = pd.DataFrame(rows)

    pd.DataFrame(mae_rows).to_csv(TABLES / "all_models_test_mae_summary.csv", index=False)
    return out


def rebuild_zone_table(metric: str) -> pd.DataFrame:
    zone_df = read_tables(ZONE_FILES)
    test = zone_df[
        (zone_df["split"] == "test")
        & (zone_df["model"].isin(MODEL_ORDER))
        & (zone_df["metric"] == metric)
    ]

    rows = []
    for model in MODEL_ORDER:
        row = {"Model": MODEL_LABELS[model]}
        sub = test[test["model"] == model]
        for h in HORIZONS:
            for zone in ZONE_ORDER:
                one = sub[(sub["horizon_min"] == h) & (sub["zone"] == zone)]
                if one.empty:
                    raise ValueError(f"Missing {metric} for {model} @ {h} min / {zone}")
                row[f"{h}m {ZONE_LABELS[zone]}"] = round(float(one["value"].iloc[0]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    write_table(rebuild_table_4_1(), "table_4_1_overall_mae_rmse_6models")
    write_table(rebuild_zone_table("mae"), "table_4_2_zone_mae_6models")
    write_table(rebuild_zone_table("rmse"), "table_4_3_zone_rmse_6models")
    print("[done] rebuilt Chapter 4 tables from tracked evaluation outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

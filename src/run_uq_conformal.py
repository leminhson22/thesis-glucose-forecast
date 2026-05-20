"""Runner for §9 Uncertainty Quantification via Conformal Prediction.

Loads the master predictions parquet, filters to the proposed model
``step6_v2_pers_resid``, calibrates Split CP and Mondrian (per-zone) CP at
alpha in {0.10, 0.20} using the val split, evaluates empirical coverage and
interval width on test, and writes:

  - outputs/tables/uq_conformal_quantiles.csv   (calibration q values)
  - outputs/tables/uq_conformal_coverage.csv    (per-horizon × per-zone)
  - outputs/tables/uq_conformal_intervals.parquet (test-set per-sample lower/upper)

The runner does not retrain the model. It works directly on saved predictions,
so it completes in seconds.

Usage
-----
::

    python src/run_uq_conformal.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uncertainty as UQ  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run conformal prediction on the proposed model.")
    parser.add_argument(
        "--predictions",
        default="outputs/tables/step6_v2_predictions.parquet",
        help="Master predictions parquet (long-form).",
    )
    parser.add_argument(
        "--model",
        default="step6_v2_pers_resid",
        help="Name of the model column to filter on.",
    )
    parser.add_argument(
        "--alpha",
        nargs="+", type=float, default=[0.10, 0.20],
        help="Mis-coverage rates. Defaults to 90 %% and 80 %% nominal coverage.",
    )
    parser.add_argument(
        "--out-tables",
        default="outputs/tables",
        help="Directory for the output CSVs / parquet.",
    )
    args = parser.parse_args()

    in_path = ROOT / args.predictions
    out_dir = ROOT / args.out_tables
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[uq] loading predictions from {in_path}")
    df = pd.read_parquet(in_path)
    print(f"[uq] loaded {len(df):,} rows; models = {sorted(df['model'].unique())}")

    runs = UQ.calibrate_and_evaluate(
        df,
        model_name=args.model,
        alphas=tuple(args.alpha),
    )

    qt = UQ.quantile_table(runs)
    qt.insert(0, "model", args.model)
    cov = UQ.coverage_summary(runs)

    # Round for readability while keeping full precision in parquet.
    qt_to_save = qt.copy()
    qt_to_save["q_mgdl"] = qt_to_save["q_mgdl"].round(3)

    cov_to_save = cov.copy()
    cov_to_save["coverage"] = (cov_to_save["coverage"] * 100).round(2)
    cov_to_save["mean_width"] = cov_to_save["mean_width"].round(2)
    cov_to_save["median_width"] = cov_to_save["median_width"].round(2)
    cov_to_save["q"] = cov_to_save["q"].round(3)
    # Re-order columns for human readability
    cov_to_save = cov_to_save[[
        "model", "method", "alpha", "horizon_min", "zone",
        "n", "coverage", "mean_width", "median_width", "q",
    ]].rename(columns={"coverage": "coverage_pct"})

    qt_path = out_dir / "uq_conformal_quantiles.csv"
    cov_path = out_dir / "uq_conformal_coverage.csv"
    qt_to_save.to_csv(qt_path, index=False)
    cov_to_save.to_csv(cov_path, index=False)
    print(f"[uq] wrote {qt_path}")
    print(f"[uq] wrote {cov_path}")

    # Build per-sample interval columns on the test split for plotting / debugging
    test_with_intervals = UQ.attach_intervals_to_predictions(
        df, runs, model_name=args.model, split="test"
    )
    int_path = out_dir / "uq_conformal_intervals.parquet"
    test_with_intervals.to_parquet(int_path, index=False)
    print(f"[uq] wrote {int_path} ({len(test_with_intervals):,} rows)")

    # Console summary
    print("\n[uq] calibration quantiles (mg/dL):")
    print(qt_to_save.to_string(index=False))
    print("\n[uq] test coverage summary (% of samples in interval):")
    print(
        cov_to_save[cov_to_save["zone"] == "all"]
        .to_string(index=False)
    )
    print("\n[uq] per-zone coverage (Mondrian, alpha=0.10):")
    sel = (
        (cov_to_save["method"] == "mondrian")
        & (np.isclose(cov_to_save["alpha"], 0.10))
        & (cov_to_save["zone"] != "all")
    )
    print(cov_to_save[sel].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply CG-EGA (Kovatchev 2004) to the master predictions parquet.

Replaces Clarke EGA as the primary clinical-safety metric per SKILL.md v2.0.
Produces three Step 5 / Step 6 artefacts:

1. ``outputs/tables/cg_ega_detail.parquet`` — per-sample point-zone,
   rate-zone, glycaemic zone and AP/BE/EP classification for every
   (model, split, participant, horizon) triplet that has enough lookback
   for the 15-min rate window.
2. ``outputs/tables/cg_ega_summary.csv`` — aggregate AP/BE/EP percentages
   per (model, split, horizon, glycaemic_zone) with sample counts.
3. ``outputs/tables/cg_ega_summary_overall.csv`` — same aggregate without
   the glycaemic-zone stratification (one row per model × split ×
   horizon, used for the headline §8 narrative).

Usage::

    python src/run_cg_ega.py            # all models present in the parquet
    python src/run_cg_ega.py --models step6_hybrid,gbm_n300

Notes:
* The master parquet must contain ``model``, ``split``, ``sample_idx``,
  ``participant_id``, ``horizon_min``, ``y_true``, ``y_pred`` columns.
* Sample_idx differences within each (model, split, participant_id,
  horizon_min) group must be 1 (val/test stride=1) for the rate
  computation to use a clean 15-min window. The function defends against
  any gaps automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from evaluate import cg_ega_from_predictions, cg_ega_summary  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

INPUT_PARQUET = TABLES_DIR / "all_models_predictions.parquet"
DETAIL_PARQUET = TABLES_DIR / "cg_ega_detail.parquet"
SUMMARY_CSV = TABLES_DIR / "cg_ega_summary.csv"
OVERALL_CSV = TABLES_DIR / "cg_ega_summary_overall.csv"


def main(models_filter: tuple[str, ...] | None) -> int:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"missing master predictions parquet: {INPUT_PARQUET}")
    t0 = time.time()
    print(f"[load] {INPUT_PARQUET.name}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"[load] rows={len(df):,}  models={sorted(df['model'].unique())}")

    if models_filter:
        keep = df["model"].isin(models_filter)
        missing = set(models_filter) - set(df["model"].unique())
        if missing:
            print(f"[warn] unknown model names ignored: {sorted(missing)}")
        df = df.loc[keep].copy()
        print(f"[filter] rows={len(df):,}  models={sorted(df['model'].unique())}")

    print("[compute] CG-EGA per (model, split, participant_id, horizon_min)")
    detail = cg_ega_from_predictions(
        df,
        group_keys=("model", "split", "participant_id", "horizon_min"),
        sort_key="sample_idx",
        rate_lag_steps=3,
        sample_step_min=C.SAMPLING_STEP_MIN,
    )
    print(f"[compute] CG-EGA labelled rows={len(detail):,}  "
          f"(dropped {len(df) - len(detail):,} due to lookback)")

    detail.to_parquet(DETAIL_PARQUET, index=False)
    print(f"[save] {DETAIL_PARQUET.name}  "
          f"({DETAIL_PARQUET.stat().st_size / 1e6:.1f} MB)")

    print("[aggregate] by (model, split, horizon, glycaemic_zone)")
    by_zone = cg_ega_summary(
        detail,
        group_cols=("model", "split", "horizon_min"),
        include_zone=True,
    )
    by_zone.to_csv(SUMMARY_CSV, index=False)
    print(f"[save] {SUMMARY_CSV.name}  rows={len(by_zone)}")

    print("[aggregate] by (model, split, horizon)  [overall, no zone strat]")
    overall = cg_ega_summary(
        detail,
        group_cols=("model", "split", "horizon_min"),
        include_zone=False,
    )
    overall.to_csv(OVERALL_CSV, index=False)
    print(f"[save] {OVERALL_CSV.name}  rows={len(overall)}")

    # Headline printout: test split, AP_pct per (model, horizon)
    print("\n========== CG-EGA test AP_pct (higher is better) ==========")
    pivot_ap = overall[overall["split"] == "test"].pivot(
        index="model", columns="horizon_min", values="AP_pct"
    ).round(2).sort_values(30, ascending=False)
    pivot_ap.columns = [f"AP_pct_{int(h)}m" for h in pivot_ap.columns]
    print(pivot_ap.to_string())

    print("\n========== CG-EGA test EP_pct (lower is better) ==========")
    pivot_ep = overall[overall["split"] == "test"].pivot(
        index="model", columns="horizon_min", values="EP_pct"
    ).round(2).sort_values(30, ascending=True)
    pivot_ep.columns = [f"EP_pct_{int(h)}m" for h in pivot_ep.columns]
    print(pivot_ep.to_string())

    print(f"\n[done] elapsed = {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", default=None,
        help="comma-separated subset of model names; default: all in parquet",
    )
    args = ap.parse_args()
    filter_set = tuple(s.strip() for s in args.models.split(",")) if args.models else None
    raise SystemExit(main(models_filter=filter_set))

# -*- coding: utf-8 -*-
"""Train an XGBoost baseline on the flattened HUPA-UCM input contract.

Same train/val/test split, same per-patient z-score scaling already
applied to data/processed/hupa_5min_sequences.npz, same evaluation
metric bundle. One booster per horizon (XGBRegressor does not produce
true multi-output regression natively).

Outputs:
    outputs/models/xgb_phase_b.joblib
    outputs/tables/xgb_predictions.parquet
    outputs/tables/xgb_summary.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
SEQ_PATH = ROOT / "data" / "processed" / "hupa_5min_sequences.npz"
SCALERS_PATH = ROOT / "outputs" / "models" / "scalers.json"
OUT_MODEL = ROOT / "outputs" / "models" / "xgb_phase_b.joblib"
OUT_PRED = ROOT / "outputs" / "tables" / "xgb_predictions.parquet"
OUT_SUMMARY = ROOT / "outputs" / "tables" / "xgb_summary.csv"


def zone_of(g: np.ndarray) -> np.ndarray:
    out = np.full(g.shape, "tir", dtype=object)
    out[g < 70] = "hypo"
    out[g > 180] = "hyper"
    return out


def main() -> None:
    print("Loading sequence bundle...")
    bundle = np.load(SEQ_PATH, allow_pickle=True)
    X_dyn = bundle["X_dynamic"]          # (N, 24, 17)
    X_stat = bundle["X_static"]          # (N, 16) -- already excludes pid col
    y = bundle["y"]                      # (N, 3)
    pid = bundle["participant_ids"]
    split = bundle["split"]
    print(f"Shapes: X_dyn={X_dyn.shape}, X_stat={X_stat.shape}, y={y.shape}")

    # Flatten dynamic into (N, 24*17) and concat with static -> (N, 424)
    N, T, F = X_dyn.shape
    X_flat = np.concatenate([X_dyn.reshape(N, T * F), X_stat], axis=1).astype(np.float32)
    print(f"Flattened: X_flat={X_flat.shape} dtype={X_flat.dtype}")

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    print(f"Split sizes: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    X_train, X_val, X_test = X_flat[train_mask], X_flat[val_mask], X_flat[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    pid_test = pid[test_mask]

    # Reload glucose scaler stats to inverse-z-score predictions
    import json
    with open(SCALERS_PATH) as f:
        scalers = json.load(f)
    glucose_mu = scalers.get("per_subject_glucose_mean", {})
    glucose_sd = scalers.get("per_subject_glucose_std", {})
    # y in the bundle is already in mg/dL (target stored unscaled per the
    # preprocessing pipeline). Confirm by checking range.
    print(f"y range: [{y.min():.1f}, {y.max():.1f}]  -> assumed mg/dL")

    models = {}
    horizons = [30, 60, 90]
    rows = []
    pred_records = []

    for h_idx, h in enumerate(horizons):
        print(f"\n=== Horizon {h} min ===")
        t0 = time.time()
        m = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            max_bin=128,
            random_state=42,
            n_jobs=2,
            early_stopping_rounds=20,
            eval_metric="mae",
        )
        m.fit(
            X_train,
            y_train[:, h_idx],
            eval_set=[(X_val, y_val[:, h_idx])],
            verbose=False,
        )
        dt = time.time() - t0
        print(f"  fit time = {dt:.1f}s, best_iteration = {m.best_iteration}")

        y_pred = m.predict(X_test)
        mae = float(np.mean(np.abs(y_test[:, h_idx] - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test[:, h_idx] - y_pred) ** 2)))
        print(f"  test MAE = {mae:.3f}  RMSE = {rmse:.3f}")

        # per-zone
        z = zone_of(y_test[:, h_idx])
        for zname in ["hypo", "tir", "hyper"]:
            zmask = z == zname
            if zmask.sum() > 0:
                zmae = float(np.mean(np.abs(y_test[zmask, h_idx] - y_pred[zmask])))
                zrmse = float(np.sqrt(np.mean((y_test[zmask, h_idx] - y_pred[zmask]) ** 2)))
                rows.append(dict(model="xgboost_n300", horizon_min=h, zone=zname, mae=zmae, rmse=zrmse, n=int(zmask.sum())))

        # overall row
        rows.append(dict(model="xgboost_n300", horizon_min=h, zone="all", mae=mae, rmse=rmse, n=int(test_mask.sum())))

        # store predictions
        N_test = test_mask.sum()
        pred_records.append(pd.DataFrame({
            "model": ["xgboost_n300"] * N_test,
            "split": ["test"] * N_test,
            "sample_idx": np.arange(N_test),
            "participant_id": pid_test,
            "horizon_min": [h] * N_test,
            "y_true": y_test[:, h_idx],
            "y_pred": y_pred,
            "abs_err": np.abs(y_test[:, h_idx] - y_pred),
            "sq_err": (y_test[:, h_idx] - y_pred) ** 2,
            "zone": z,
        }))

        models[h] = m

    # save
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, OUT_MODEL)
    pd.concat(pred_records, ignore_index=True).to_parquet(OUT_PRED, index=False)
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved:\n  model -> {OUT_MODEL}\n  predictions -> {OUT_PRED}\n  summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()

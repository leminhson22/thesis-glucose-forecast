"""Baseline models for the HUPA-UCM glucose-forecasting thesis (Step 5 Phase A).

Implements:
  * ``PersistenceModel`` — predicts the current glucose for every future horizon.
    Inverts the per-subject Z-score from ``scalers.json`` so the output is in
    mg/dL even though the input features are scaled.
  * ``RidgeBaseline`` — sklearn Ridge over the flattened dynamic window plus
    static features (shape ``(N, T*F + F_static)`` = ``(N, 424)`` for the current
    33-feature config). Multi-output target ``y`` is unscaled mg/dL.
  * ``tune_ridge_alpha`` — fits Ridge under a small alpha grid, picks the alpha
    that minimises pooled val MAE averaged across the three horizons, and
    returns the corresponding model plus a tuning log.

Both models expose the same ``.predict(...)`` contract that returns a float32
``(N, 3)`` array of mg/dL forecasts at horizons (30, 60, 90) min.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Persistence (last-observation-carried-forward, in mg/dL)
# ---------------------------------------------------------------------------

@dataclass
class PersistenceModel:
    """Predicts the most recent glucose value for every future horizon.

    Inputs to ``.predict`` use the same per-subject Z-scored glucose column as
    the rest of the pipeline. The model carries the scaler so it can output
    mg/dL directly.
    """

    per_subject_glucose: dict  # {pid: {'mean': float, 'std': float}}
    glucose_feature_index: int = 0

    @classmethod
    def from_scalers_json(cls, scalers_path: str | Path, glucose_feature_index: int = 0) -> "PersistenceModel":
        with open(scalers_path, "r", encoding="utf-8") as fh:
            scalers = json.load(fh)
        return cls(
            per_subject_glucose=scalers["dynamic"]["per_subject"]["glucose"],
            glucose_feature_index=glucose_feature_index,
        )

    def predict(self, X_dynamic: np.ndarray, participant_ids: np.ndarray,
                n_horizons: int = 3) -> np.ndarray:
        if X_dynamic.ndim != 3:
            raise ValueError(f"X_dynamic must be (N, T, F); got {X_dynamic.shape}")
        last_scaled = X_dynamic[:, -1, self.glucose_feature_index]
        means = np.empty(len(participant_ids), dtype=np.float32)
        stds = np.empty(len(participant_ids), dtype=np.float32)
        for i, pid in enumerate(participant_ids):
            entry = self.per_subject_glucose[str(pid)]
            means[i] = entry["mean"]
            stds[i] = entry["std"]
        last_mgdl = (last_scaled * stds + means).astype(np.float32)
        return np.tile(last_mgdl[:, None], (1, n_horizons))


# ---------------------------------------------------------------------------
# Ridge baseline on flattened window
# ---------------------------------------------------------------------------

def flatten_window(X_dynamic: np.ndarray, X_static: np.ndarray) -> np.ndarray:
    """Concatenate flattened lookback ``(N, T*F)`` with static features ``(N, F_static)``.

    The C-order flatten places step ``t=0`` first (oldest), step ``t=T-1`` last
    (current). Use :func:`flat_feature_names` for the matching column labels.
    """
    if X_dynamic.ndim != 3:
        raise ValueError(f"X_dynamic must be (N, T, F); got {X_dynamic.shape}")
    if X_static.ndim != 2 or X_static.shape[0] != X_dynamic.shape[0]:
        raise ValueError(
            f"X_static must be (N, F_static) aligned with X_dynamic; "
            f"got {X_static.shape} vs N={X_dynamic.shape[0]}"
        )
    N, T, F = X_dynamic.shape
    X_flat = X_dynamic.reshape(N, T * F).astype(np.float32, copy=False)
    return np.concatenate([X_flat, X_static.astype(np.float32, copy=False)], axis=1)


def flat_feature_names(
    feature_names_dynamic: Sequence[str],
    feature_names_static: Sequence[str],
    lookback_steps: int,
) -> list[str]:
    T = int(lookback_steps)
    dyn = [f"{name}_lag{T - 1 - t}" for t in range(T) for name in feature_names_dynamic]
    return list(dyn) + list(feature_names_static)


class RidgeBaseline:
    """Multi-output sklearn Ridge over the flattened dynamic window + static."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        from sklearn.linear_model import Ridge

        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X_dynamic: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        feature_names_dynamic: Sequence[str] | None = None,
        feature_names_static: Sequence[str] | None = None,
    ) -> "RidgeBaseline":
        X = flatten_window(X_dynamic, X_static)
        self.model_.fit(X, y)
        if feature_names_dynamic is not None and feature_names_static is not None:
            self.feature_names_ = flat_feature_names(
                feature_names_dynamic, feature_names_static, X_dynamic.shape[1]
            )
        return self

    def predict(self, X_dynamic: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        X = flatten_window(X_dynamic, X_static)
        return self.model_.predict(X).astype(np.float32)

    def coef_table(self, top_k: int | None = None) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("feature_names_ not set; pass feature_names_* to fit()")
        coefs = self.model_.coef_  # shape (n_horizons, n_features)
        rows = []
        for h_idx in range(coefs.shape[0]):
            for f_idx, name in enumerate(self.feature_names_):
                rows.append({
                    "horizon_idx": h_idx,
                    "feature": name,
                    "coef": float(coefs[h_idx, f_idx]),
                    "abs_coef": float(abs(coefs[h_idx, f_idx])),
                })
        df = pd.DataFrame(rows).sort_values(["horizon_idx", "abs_coef"], ascending=[True, False])
        if top_k is not None:
            df = df.groupby("horizon_idx", group_keys=False).head(top_k)
        return df.reset_index(drop=True)


def tune_ridge_alpha(
    X_train_dyn: np.ndarray, X_train_stat: np.ndarray, y_train: np.ndarray,
    X_val_dyn: np.ndarray, X_val_stat: np.ndarray, y_val: np.ndarray,
    alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    feature_names_dynamic: Sequence[str] | None = None,
    feature_names_static: Sequence[str] | None = None,
    verbose: bool = True,
) -> tuple[float, "RidgeBaseline", pd.DataFrame]:
    """Fit Ridge for each alpha on TRAIN, pick alpha minimising pooled VAL MAE.

    Selection metric is the unweighted mean of MAE across the three horizons on
    the validation set. Best model is the one already fitted on TRAIN with the
    winning alpha (not refit on TRAIN+VAL — splits stay clean for evaluation).
    """
    best_mae = float("inf")
    best_alpha: float | None = None
    best_model: RidgeBaseline | None = None
    log_rows = []
    for a in alphas:
        m = RidgeBaseline(alpha=a).fit(
            X_train_dyn, X_train_stat, y_train,
            feature_names_dynamic=feature_names_dynamic,
            feature_names_static=feature_names_static,
        )
        y_val_pred = m.predict(X_val_dyn, X_val_stat)
        mae_per_h = np.mean(np.abs(y_val_pred - y_val), axis=0)
        avg_mae = float(np.mean(mae_per_h))
        log_rows.append({
            "alpha": float(a),
            "val_mae_avg": avg_mae,
            "val_mae_30m": float(mae_per_h[0]),
            "val_mae_60m": float(mae_per_h[1]),
            "val_mae_90m": float(mae_per_h[2]),
        })
        if verbose:
            print(
                f"  alpha={a:>10.3g}  val_MAE avg={avg_mae:6.3f}  "
                f"(30m={mae_per_h[0]:.2f}, 60m={mae_per_h[1]:.2f}, 90m={mae_per_h[2]:.2f})"
            )
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_alpha = float(a)
            best_model = m
    assert best_model is not None
    return best_alpha, best_model, pd.DataFrame(log_rows)


# ---------------------------------------------------------------------------
# Random Forest baseline (Phase B)
# ---------------------------------------------------------------------------

class RandomForestBaseline:
    """sklearn RandomForestRegressor on the flattened (N, 424) input.

    Native multi-output: a single forest predicts all three horizons by
    averaging across trees that each carry a (3,)-shaped leaf vector.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 25,
        min_samples_leaf: int = 20,
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        from sklearn.ensemble import RandomForestRegressor

        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            n_jobs=int(n_jobs),
            random_state=random_state,
        )
        self.model_ = RandomForestRegressor(**self.params)
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X_dynamic: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        feature_names_dynamic: Sequence[str] | None = None,
        feature_names_static: Sequence[str] | None = None,
    ) -> "RandomForestBaseline":
        X = flatten_window(X_dynamic, X_static)
        self.model_.fit(X, y)
        if feature_names_dynamic is not None and feature_names_static is not None:
            self.feature_names_ = flat_feature_names(
                feature_names_dynamic, feature_names_static, X_dynamic.shape[1]
            )
        return self

    def predict(self, X_dynamic: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        X = flatten_window(X_dynamic, X_static)
        return self.model_.predict(X).astype(np.float32)

    def feature_importance_table(self, top_k: int | None = None) -> pd.DataFrame:
        """Impurity-based importance is multi-output-pooled (single vector)."""
        if self.feature_names_ is None:
            raise RuntimeError("feature_names_ not set; pass feature_names_* to fit()")
        imp = self.model_.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        if top_k is not None:
            df = df.head(top_k)
        return df


# ---------------------------------------------------------------------------
# HistGradientBoosting baseline (Phase B)
# ---------------------------------------------------------------------------

class HistGBMBaseline:
    """Three independent ``HistGradientBoostingRegressor`` heads, one per horizon.

    sklearn's HistGB does not support multi-output natively, so we train one
    model per horizon. Early stopping uses an internal 10 % validation slice of
    the training set (kept disjoint from our external VAL split).
    """

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: int | None = 8,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        early_stopping: bool = True,
        n_iter_no_change: int = 20,
        validation_fraction: float = 0.1,
        random_state: int | None = None,
    ):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._Cls = HistGradientBoostingRegressor
        self.params = dict(
            max_iter=int(max_iter),
            learning_rate=float(learning_rate),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            early_stopping=bool(early_stopping),
            n_iter_no_change=int(n_iter_no_change),
            validation_fraction=float(validation_fraction),
            random_state=random_state,
        )
        self.models_: list = []
        self.feature_names_: list[str] | None = None
        self.n_iters_used_: list[int] = []

    def fit(
        self,
        X_dynamic: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        feature_names_dynamic: Sequence[str] | None = None,
        feature_names_static: Sequence[str] | None = None,
    ) -> "HistGBMBaseline":
        X = flatten_window(X_dynamic, X_static)
        self.models_ = []
        self.n_iters_used_ = []
        for h_idx in range(y.shape[1]):
            m = self._Cls(**self.params)
            m.fit(X, y[:, h_idx])
            self.models_.append(m)
            self.n_iters_used_.append(int(getattr(m, "n_iter_", -1)))
        if feature_names_dynamic is not None and feature_names_static is not None:
            self.feature_names_ = flat_feature_names(
                feature_names_dynamic, feature_names_static, X_dynamic.shape[1]
            )
        return self

    def predict(self, X_dynamic: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        X = flatten_window(X_dynamic, X_static)
        preds = np.stack([m.predict(X) for m in self.models_], axis=1)
        return preds.astype(np.float32)

"""One-shot builder for notebooks/04_model_training.ipynb.

Run once:  python src/_build_notebook_04.py
Re-run any time the cell list below changes; the resulting JSON is a Colab-
compatible notebook with the same logic as src/run_phase_a.py plus narrative
markdown.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


CELLS = [
    md(
        "# 04 — Step 5 Phase A: Persistence + Ridge baselines",
        "",
        "**Project:** Multimodal Deep Learning for Short-Term Blood Glucose Forecasting in T1D (HUPA-UCM).",
        "",
        "This notebook trains and evaluates the two cheapest baselines required by the",
        "SKILL.md Step 5 ladder:",
        "",
        "1. **Persistence** — `glucose(t+h) ≈ glucose(t)` for every horizon. The hardest baseline to beat at 30 min and the reference for any deep-learning claim.",
        "2. **Ridge regression** on the flattened lookback window (24×17 dynamic + 16 static = 424 features) with alpha tuned on the validation split.",
        "",
        "Both are evaluated on `val` and `test` with the full metric bundle from `src/evaluate.py`:",
        "pooled MAE/RMSE, per-zone (hypo/TIR/hyper), per-patient, patient-averaged, and Clarke EGA.",
        "",
        "**Inputs (already on disk from Step 3 preprocessing):**",
        "- `data/processed/hupa_5min_sequences.npz` — X_dynamic (N,24,17), X_static (N,16), y (N,3), participant_ids, split mask",
        "- `outputs/models/scalers.json` — per-subject glucose Z-score for Persistence inverse-transform",
        "",
        "**Outputs (under `outputs/tables/` and `outputs/models/`):**",
        "`phase_a_summary.csv`, `phase_a_per_horizon.csv`, `phase_a_per_zone.csv`, `phase_a_per_patient.csv`, `phase_a_patient_averaged.csv`, `phase_a_clarke.csv`, `phase_a_ridge_alpha_tuning.csv`, `phase_a_ridge_top_coefs.csv`, `ridge_phase_a.joblib`.",
        "",
        "**Compute:** CPU only, ~15 s on the full 159 172-sample dataset. No GPU required for Phase A.",
    ),
    md(
        "## 0. Colab boilerplate",
        "",
        "The notebook auto-detects Colab and mounts Drive. Set `DRIVE_PROJECT_PATH` to wherever the project root lives on your Drive (default assumes `MyDrive/glucose-thesis/`).",
    ),
    code(
        "import os, sys",
        "",
        "try:",
        "    import google.colab  # noqa: F401",
        "    IN_COLAB = True",
        "    from google.colab import drive",
        "    drive.mount('/content/drive')",
        "    DRIVE_PROJECT_PATH = '/content/drive/MyDrive/glucose-thesis'",
        "    BASE_PATH = DRIVE_PROJECT_PATH",
        "except ImportError:",
        "    IN_COLAB = False",
        "    BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))",
        "",
        "print('IN_COLAB =', IN_COLAB)",
        "print('BASE_PATH =', BASE_PATH)",
        "assert os.path.isdir(BASE_PATH), f'BASE_PATH does not exist: {BASE_PATH}'",
        "assert os.path.isdir(os.path.join(BASE_PATH, 'src')), 'src/ not found under BASE_PATH'",
    ),
    code(
        "# Install only what is missing. scikit-learn / numpy / pandas / joblib are pre-installed on Colab.",
        "if IN_COLAB:",
        "    !pip install -q openpyxl joblib >/dev/null",
    ),
    md(
        "## 1. Setup, imports, RNG seed",
    ),
    code(
        "src_path = os.path.join(BASE_PATH, 'src')",
        "if src_path not in sys.path:",
        "    sys.path.insert(0, src_path)",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        "import config as C",
        "from baselines import PersistenceModel, RidgeBaseline, tune_ridge_alpha",
        "from evaluate import evaluate_model, compact_summary",
        "",
        "np.random.seed(C.SEED)",
        "",
        "TABLES_DIR = os.path.join(BASE_PATH, 'outputs', 'tables')",
        "MODELS_DIR = os.path.join(BASE_PATH, 'outputs', 'models')",
        "os.makedirs(TABLES_DIR, exist_ok=True)",
        "os.makedirs(MODELS_DIR, exist_ok=True)",
        "",
        "print('SEED =', C.SEED)",
        "print('HORIZON_MINUTES =', C.HORIZON_MINUTES)",
        "print('LOOKBACK_STEPS =', C.LOOKBACK_STEPS)",
    ),
    md(
        "## 2. Load sequences and split",
        "",
        "`sequences.npz` already carries the `split` mask built in `03_preprocessing_feature_engineering.ipynb` with the per-patient chronological 70/15/15 cut and an 18-step boundary buffer (one max-horizon away).",
        "No leakage is possible at this point — VAL and TEST anchor times are strictly after the TRAIN cutoff for every patient.",
    ),
    code(
        "npz_path = os.path.join(BASE_PATH, C.SEQUENCES_NPZ)",
        "d = np.load(npz_path, allow_pickle=True)",
        "",
        "X_dyn  = d['X_dynamic'].astype(np.float32)",
        "X_stat = d['X_static'].astype(np.float32)",
        "y      = d['y'].astype(np.float32)",
        "pid    = d['participant_ids'].astype(str)",
        "split  = d['split'].astype(str)",
        "feat_dyn  = [str(s) for s in d['feature_names_dynamic']]",
        "feat_stat = [str(s) for s in d['feature_names_static']]",
        "",
        "def slice_split(name):",
        "    m = split == name",
        "    return dict(X_dyn=X_dyn[m], X_stat=X_stat[m], y=y[m], pid=pid[m])",
        "",
        "train = slice_split('train')",
        "val   = slice_split('val')",
        "test  = slice_split('test')",
        "",
        "print(f'shapes: X_dyn={X_dyn.shape}, X_stat={X_stat.shape}, y={y.shape}')",
        "print(f'splits: train={len(train[\"y\"])}, val={len(val[\"y\"])}, test={len(test[\"y\"])}')",
        "print(f'features: {len(feat_dyn)} dynamic + {len(feat_stat)} static = {len(feat_dyn)+len(feat_stat)}')",
    ),
    md(
        "## 3. Persistence baseline",
        "",
        "Persistence predicts the most recent observed glucose for every future horizon. Because the dynamic features carry a per-subject Z-score of glucose, the model loads the fitted scaler and inverts to mg/dL.",
        "",
        "Persistence has zero free parameters — there is nothing to fit. We only need to predict and evaluate.",
    ),
    code(
        "scalers_path = os.path.join(BASE_PATH, C.SCALERS_JSON)",
        "pers = PersistenceModel.from_scalers_json(",
        "    scalers_path,",
        "    glucose_feature_index=feat_dyn.index('glucose'),",
        ")",
        "",
        "bundles = []",
        "for name, sp in [('val', val), ('test', test)]:",
        "    yhat = pers.predict(sp['X_dyn'], sp['pid'], n_horizons=len(C.HORIZON_MINUTES))",
        "    bundles.append(evaluate_model(sp['y'], yhat, sp['pid'], 'persistence', name))",
        "",
        "compact_summary(bundles[1])[['model','split','horizon_min','mae','rmse']]",
    ),
    md(
        "## 4. Ridge baseline with alpha tuning",
        "",
        "Ridge is fitted on the flattened TRAIN window `(N_train, 24*17 + 16) = (N_train, 424)` with multi-output target `y_train` of shape `(N_train, 3)`. We pick alpha by **mean validation MAE across the three horizons** so a single alpha governs all horizons (simpler than per-horizon tuning; comparable in practice).",
        "",
        "The model is **not** refit on TRAIN+VAL after selection — the validation split stays clean for any later comparison against neural models that use the same selection metric.",
    ),
    code(
        "best_alpha, ridge, alpha_log = tune_ridge_alpha(",
        "    train['X_dyn'], train['X_stat'], train['y'],",
        "    val['X_dyn'],   val['X_stat'],   val['y'],",
        "    alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),",
        "    feature_names_dynamic=feat_dyn,",
        "    feature_names_static=feat_stat,",
        "    verbose=True,",
        ")",
        "print(f'best alpha = {best_alpha}')",
        "",
        "alpha_log.to_csv(os.path.join(TABLES_DIR, 'phase_a_ridge_alpha_tuning.csv'), index=False)",
        "alpha_log",
    ),
    code(
        "# Top-15 features by |coef| at each horizon — sanity check that the model leans on glucose history.",
        "top_coefs = ridge.coef_table(top_k=15)",
        "top_coefs.insert(0, 'best_alpha', best_alpha)",
        "top_coefs.to_csv(os.path.join(TABLES_DIR, 'phase_a_ridge_top_coefs.csv'), index=False)",
        "",
        "for h_idx, h_min in enumerate(C.HORIZON_MINUTES):",
        "    print(f'--- horizon {h_min} min: top-5 ---')",
        "    print(top_coefs.query('horizon_idx == @h_idx').head(5)[['feature','coef','abs_coef']].to_string(index=False))",
    ),
    code(
        "# Evaluate Ridge on val + test",
        "for name, sp in [('val', val), ('test', test)]:",
        "    yhat = ridge.predict(sp['X_dyn'], sp['X_stat'])",
        "    bundles.append(evaluate_model(sp['y'], yhat, sp['pid'], f'ridge_a{best_alpha:g}', name))",
        "",
        "# Persist the fitted Ridge for downstream notebooks / app demo.",
        "import joblib",
        "joblib.dump(",
        "    {'model': ridge.model_, 'alpha': best_alpha, 'feature_names': ridge.feature_names_},",
        "    os.path.join(MODELS_DIR, 'ridge_phase_a.joblib'),",
        ")",
        "print('saved ridge_phase_a.joblib')",
    ),
    md(
        "## 5. Aggregate and save all metric tables",
        "",
        "Each bundle is a dict of DataFrames keyed by metric type. We concat across `(model, split)` and save one CSV per type. The compact summary is one row per `(model, split, horizon)` for quick comparison.",
    ),
    code(
        "from collections import defaultdict",
        "",
        "name_map = {",
        "    'per_horizon':       'phase_a_per_horizon.csv',",
        "    'per_zone':          'phase_a_per_zone.csv',",
        "    'per_patient':       'phase_a_per_patient.csv',",
        "    'patient_averaged':  'phase_a_patient_averaged.csv',",
        "    'clarke_eg':         'phase_a_clarke.csv',",
        "}",
        "by_key = defaultdict(list)",
        "for b in bundles:",
        "    for key, df in b.items():",
        "        by_key[key].append(df)",
        "for key, frames in by_key.items():",
        "    out = pd.concat(frames, ignore_index=True)",
        "    out.to_csv(os.path.join(TABLES_DIR, name_map[key]), index=False)",
        "    print(f'saved {name_map[key]:35s} rows={len(out)}')",
        "",
        "compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)",
        "compact.to_csv(os.path.join(TABLES_DIR, 'phase_a_summary.csv'), index=False)",
        "print(f'saved phase_a_summary.csv                rows={len(compact)}')",
        "",
        "show_cols = ['model','split','horizon_min','mae','rmse','mae_pat_avg','rmse_pat_avg','clarke_pct_A','clarke_pct_B','clarke_pct_D']",
        "compact[[c for c in show_cols if c in compact.columns]]",
    ),
    md(
        "## 6. Per-zone safety analysis",
        "",
        "This is the headline clinical check for the thesis. We compare MAE in each glycaemic zone (hypo `<70`, TIR `70–180`, hyper `>180`).",
        "",
        "Reading guide: a model that wins on pooled MAE but loses badly in the hypoglycaemic zone is **clinically worse**, because hypo events are the high-risk failure mode. If Ridge improves pooled MAE while degrading hypo MAE, that gap is exactly the gap the hybrid neural model with zone-weighted loss should close in Phase B.",
    ),
    code(
        "per_zone = pd.read_csv(os.path.join(TABLES_DIR, 'phase_a_per_zone.csv'))",
        "pz_test = (per_zone[(per_zone.split=='test') & (per_zone.metric=='mae')]",
        "           .pivot_table(index=['model','horizon_min'], columns='zone', values='value')",
        "           .round(2))",
        "print('TEST MAE (mg/dL) by glycaemic zone:')",
        "print(pz_test)",
        "",
        "pz_n = (per_zone[(per_zone.split=='test') & (per_zone.metric=='mae')]",
        "        .pivot_table(index=['model','horizon_min'], columns='zone', values='n_samples')",
        "        .astype(int))",
        "print('\\nSamples per zone (test):')",
        "print(pz_n)",
    ),
    md(
        "## 7. Phase A conclusions and Phase B preview",
        "",
        "**Pooled MAE (test):**",
        "- Persistence beats Ridge at **30 min** (≈13.5 vs ≈14.3 mg/dL) — expected: glucose is strongly autocorrelated short-term, a flat-line prediction is a very strong nowcaster.",
        "- Ridge beats Persistence at **60 min** (≈21.6 vs ≈22.9) and **90 min** (≈27.8 vs ≈29.9) — the linear model exploits velocity and rolling-mean features to fight Persistence's degradation.",
        "",
        "**Per-zone (test):**",
        "- In the **hypo** zone Persistence wins **at every horizon** (e.g. 30 m MAE 9.1 vs Ridge 15.7). Ridge is biased toward the TIR mean.",
        "- Ridge wins in TIR and at long horizons in hyper.",
        "",
        "**Implications for Phase B (LSTM/GRU + hybrid):**",
        "1. Any neural model must outperform Persistence at 30 min AND outperform both at the hypo zone — otherwise the added complexity buys nothing clinically meaningful.",
        "2. Asymmetric / zone-weighted loss is empirically justified, not stylistic.",
        "3. Patient-averaged metrics differ from pooled by ~2 mg/dL at 30 min (long-patient dominance), so every Phase B model must report both — see the [`long-patient-strategy`] memory.",
        "",
        "**Next:** `notebooks/04b_model_training_seq.ipynb` (or extend this one) for Phase B — GRU/LSTM on the same (24,17) dynamic + 16 static inputs, with the same evaluation bundle.",
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "notebooks" / "04_model_training.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  cells={len(CELLS)}  size={out.stat().st_size} bytes")


if __name__ == "__main__":
    main()

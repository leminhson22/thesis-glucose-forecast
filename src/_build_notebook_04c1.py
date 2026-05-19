"""One-shot builder for notebooks/04c1_phase_c1_lstm_gru.ipynb.

Run once:  python src/_build_notebook_04c1.py

Mirrors _build_notebook_04b.py for Phase C.1 (LSTM + GRU recurrent baselines
with the same X_dynamic + X_static input contract as the tree baselines).
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
        "# 04c1 — Step 5 Phase C.1: LSTM + GRU recurrent baselines",
        "",
        "**Project:** Multimodal Deep Learning for Short-Term Blood Glucose Forecasting in T1D (HUPA-UCM).",
        "",
        "Phase C.1 trains two recurrent baselines — LSTM and GRU — under the same input contract as Phase A/B:",
        "",
        "* `X_dynamic`: `(N, 24, 17)` — 17 dynamic features over a 24-step (120-minute) lookback window.",
        "* `X_static`:  `(N, 16)` — 16 patient-level static features (clinical metadata + train-only derived stats).",
        "* `y`:         `(N, 3)` — mg/dL targets at 30 / 60 / 90 min.",
        "",
        "**Apples-to-apples rationale.** Every Phase A/B baseline (Ridge, Random Forest, HistGradientBoosting) consumes the *flattened* concatenation of `X_dynamic` and `X_static` (`src/baselines.py::flatten_window`, yielding `(N, 24·17 + 16) = (N, 424)`). To make Phase C numbers directly comparable, the LSTM/GRU encoder ingests `X_dynamic` (preserving the sequential structure that the tree baselines flattened away) and a separate dense MLP ingests `X_static`; the two embeddings are concatenated before the multi-horizon output head.",
        "",
        "**Loss.** Vanilla `MultiHorizonMSE` (mean squared error over batch and horizons). Phase C.2 will replace this with asymmetric / zone-weighted variants designed for the hypo zone.",
        "",
        "**Compute.** GPU recommended — Colab T4 is sufficient. Local CPU is feasible for smoke-testing (`DEBUG = True`) but slow for full runs (~30+ minutes per model). Set `DEBUG = True` in cell 4 to run a 5k-train / 2k-val / 2k-test smoke test in ~30s on CPU.",
    ),
    md(
        "## 0. Colab boilerplate",
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
    ),
    code(
        "# PyTorch is pre-installed on Colab (GPU build). Local users: install torch matching their setup.",
        "# This cell is a no-op on Colab if torch is already importable.",
        "if IN_COLAB:",
        "    try:",
        "        import torch  # noqa: F401",
        "    except ImportError:",
        "        !pip install -q torch >/dev/null  # noqa",
    ),
    md(
        "## 1. Setup, imports, DEBUG switch",
    ),
    code(
        "src_path = os.path.join(BASE_PATH, 'src')",
        "if src_path not in sys.path:",
        "    sys.path.insert(0, src_path)",
        "",
        "import time",
        "from collections import defaultdict",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "",
        "import config as C",
        "from datasets import load_npz_splits, build_dataloaders",
        "from losses import MultiHorizonMSE",
        "from models import lstm_regressor, gru_regressor, count_parameters",
        "from train import TrainConfig, train_model, get_device, seed_everything",
        "from evaluate import compact_summary",
        "",
        "seed_everything(C.SEED)",
        "",
        "TABLES_DIR = os.path.join(BASE_PATH, 'outputs', 'tables')",
        "LOGS_DIR   = os.path.join(BASE_PATH, 'outputs', 'logs')",
        "MODELS_DIR = os.path.join(BASE_PATH, 'outputs', 'models')",
        "os.makedirs(TABLES_DIR, exist_ok=True)",
        "os.makedirs(LOGS_DIR, exist_ok=True)",
        "os.makedirs(MODELS_DIR, exist_ok=True)",
        "",
        "# Flip DEBUG = True for a 5k/2k/2k subsample × 2 epochs smoke test (~30s CPU).",
        "# Full run uses ~30 epochs on the complete train set with early stopping at patience 10.",
        "DEBUG = False",
        "device = get_device()",
        "print(f'DEBUG = {DEBUG}, device = {device}, torch = {torch.__version__}')",
    ),
    md(
        "## 2. Load sequences and (optionally) subsample",
        "",
        "Same `.npz` file as Phase A/B — the `split` mask carries the per-patient chronological 70/15/15 cut with an 18-step boundary buffer. The `build_dataloaders` helper wraps each split in a `HUPASequenceDataset` and returns `DataLoader`s; the train loader shuffles, val and test do not (so participant IDs align row-by-row with predictions).",
    ),
    code(
        "npz_path = os.path.join(BASE_PATH, C.SEQUENCES_NPZ)",
        "splits = load_npz_splits(npz_path)",
        "",
        "def subsample(sp, n_cap, seed=C.SEED):",
        "    rng = np.random.default_rng(seed)",
        "    n = sp['y'].shape[0]",
        "    if n_cap <= 0 or n <= n_cap:",
        "        return sp",
        "    idx = np.sort(rng.choice(n, size=n_cap, replace=False))",
        "    return {k: v[idx] for k, v in sp.items()}",
        "",
        "if DEBUG:",
        "    splits['train'] = subsample(splits['train'], 5000)",
        "    splits['val']   = subsample(splits['val'],   2000)",
        "    splits['test']  = subsample(splits['test'],  2000)",
        "    EPOCHS, BATCH = 2, 64",
        "else:",
        "    EPOCHS, BATCH = 30, 128",
        "",
        "feat_dyn  = splits['feat_dyn']",
        "feat_stat = splits['feat_stat']",
        "n_dynamic = len(feat_dyn)",
        "n_static  = len(feat_stat)",
        "sizes = {k: splits[k]['y'].shape[0] for k in ('train', 'val', 'test')}",
        "print(f'sizes={sizes}  n_dyn={n_dynamic}  n_stat={n_static}  EPOCHS={EPOCHS}  BATCH={BATCH}')",
        "",
        "loaders = build_dataloaders(splits, batch_size=BATCH, num_workers=0, seed=C.SEED, pin_memory=(device.type == 'cuda'))",
    ),
    md(
        "## 3. Train LSTM",
        "",
        "Architecture (default config in `src/models.py::DEFAULTS`): 2-layer LSTM with hidden dim 64, dropout 0.2 between layers, then static MLP → embedding dim 32 → concat → dense(64) → linear(3). Total ~62k parameters.",
        "",
        "Optimizer Adam, learning rate 1e-3, weight decay 1e-5, gradient clip at 1.0, ReduceLROnPlateau patience 5 (factor 0.5, min 1e-6), early stopping patience 10 epochs on **patient-averaged val MAE** (the early-stopping target chosen so HUPA0027 alone cannot dominate model selection — see the long-patient-strategy memory).",
    ),
    code(
        "t0 = time.time()",
        "lstm = lstm_regressor(n_dynamic=n_dynamic, n_static=n_static)",
        "print(f'LSTM params = {count_parameters(lstm):,}')",
        "cfg = TrainConfig(",
        "    epochs=EPOCHS,",
        "    early_stopping_patience=min(10, max(2, EPOCHS // 3)),",
        "    lr_scheduler_patience=min(5, max(1, EPOCHS // 5)),",
        ")",
        "lstm_result = train_model(",
        "    model=lstm,",
        "    loaders=loaders,",
        "    loss_fn=MultiHorizonMSE(reduction='mean'),",
        "    cfg=cfg,",
        "    run_tag=f\"lstm_phase_c1{'_debug' if DEBUG else ''}\",",
        "    logs_dir=LOGS_DIR,",
        "    models_dir=MODELS_DIR,",
        "    verbose=True,",
        ")",
        "print(f'\\n[lstm] {time.time() - t0:.1f}s; best epoch={lstm_result[\"best_epoch\"]}; '",
        "      f'best val pat-avg MAE={lstm_result[\"best_val_pat_avg_mae\"]:.3f}')",
    ),
    md(
        "## 4. Train GRU",
        "",
        "Same architecture and training protocol, only the recurrent cell changes. GRU has fewer parameters (~49k) and is typically faster per epoch.",
    ),
    code(
        "t0 = time.time()",
        "gru = gru_regressor(n_dynamic=n_dynamic, n_static=n_static)",
        "print(f'GRU params = {count_parameters(gru):,}')",
        "gru_result = train_model(",
        "    model=gru,",
        "    loaders=loaders,",
        "    loss_fn=MultiHorizonMSE(reduction='mean'),",
        "    cfg=cfg,",
        "    run_tag=f\"gru_phase_c1{'_debug' if DEBUG else ''}\",",
        "    logs_dir=LOGS_DIR,",
        "    models_dir=MODELS_DIR,",
        "    verbose=True,",
        ")",
        "print(f'\\n[gru] {time.time() - t0:.1f}s; best epoch={gru_result[\"best_epoch\"]}; '",
        "      f'best val pat-avg MAE={gru_result[\"best_val_pat_avg_mae\"]:.3f}')",
    ),
    md(
        "## 5. Aggregate val + test metric bundles",
        "",
        "Same evaluation bundle as Phase A/B (`src/evaluate.py`): per-horizon, per-zone, per-patient, patient-averaged, Clarke EGA. Saved under `outputs/tables/phase_c1_*.csv`.",
    ),
    code(
        "bundles = []",
        "for r in (lstm_result, gru_result):",
        "    bundles.append(r['final']['val']['bundle'])",
        "    bundles.append(r['final']['test']['bundle'])",
        "",
        "name_map = {",
        "    'per_horizon':      'phase_c1_per_horizon.csv',",
        "    'per_zone':         'phase_c1_per_zone.csv',",
        "    'per_patient':      'phase_c1_per_patient.csv',",
        "    'patient_averaged': 'phase_c1_patient_averaged.csv',",
        "    'clarke_eg':        'phase_c1_clarke.csv',",
        "}",
        "by_key = defaultdict(list)",
        "for b in bundles:",
        "    for key, df in b.items():",
        "        by_key[key].append(df)",
        "for key, frames in by_key.items():",
        "    out = pd.concat(frames, ignore_index=True)",
        "    out.to_csv(os.path.join(TABLES_DIR, name_map[key]), index=False)",
        "    print(f'saved {name_map[key]:34s} rows={len(out)}')",
        "",
        "compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)",
        "compact.to_csv(os.path.join(TABLES_DIR, 'phase_c1_summary.csv'), index=False)",
        "show = ['model','split','horizon_min','mae','rmse','mae_pat_avg','rmse_pat_avg','clarke_pct_A','clarke_pct_D']",
        "compact[[c for c in show if c in compact.columns]]",
    ),
    md(
        "## 6. Comparison vs GBM-300 (the best Phase B baseline)",
        "",
        "Phase B's best baseline was HistGradientBoosting with `max_iter = 300` and pooled test MAE 10.40 / 19.77 / 26.27 mg/dL at 30 / 60 / 90 min. We tabulate the LSTM and GRU test MAE side-by-side with these numbers.",
    ),
    code(
        "GBM_300_TEST = {30: 10.40, 60: 19.77, 90: 26.27}",
        "rows = []",
        "for _, r in compact[compact['split'] == 'test'].iterrows():",
        "    h = int(r['horizon_min'])",
        "    mae = float(r['mae'])",
        "    delta = mae - GBM_300_TEST[h]",
        "    rel = 100.0 * delta / GBM_300_TEST[h]",
        "    rows.append({",
        "        'model': r['model'],",
        "        'horizon_min': h,",
        "        'mae_test': round(mae, 3),",
        "        'mae_gbm_300': GBM_300_TEST[h],",
        "        'mae_delta': round(delta, 3),",
        "        'mae_rel_pct': round(rel, 2),",
        "    })",
        "vs_gbm = pd.DataFrame(rows)",
        "vs_gbm.to_csv(os.path.join(TABLES_DIR, 'phase_c1_vs_gbm_300.csv'), index=False)",
        "vs_gbm",
    ),
    md(
        "## 7. Per-zone test MAE (hypo / TIR / hyper)",
        "",
        "The clinically critical check (mirrors Phase B §7). The Phase A/B pattern was: increasing model capacity along the tree ladder reduces pooled MAE but does **not** close the hypo-zone gap. We ask whether recurrent encoders without a loss-function change close that gap. Phase C.1 expectation: pooled MAE may match or beat GBM, hypo MAE likely still worse than Persistence — that expectation is the motivation for C.2's asymmetric/zone-weighted loss.",
    ),
    code(
        "pz = pd.concat([b['per_zone'] for b in bundles], ignore_index=True)",
        "pz_test_mae = (pz[(pz['split'] == 'test') & (pz['metric'] == 'mae')]",
        "               .pivot_table(index=['model', 'horizon_min'], columns='zone', values='value')",
        "               .round(2))",
        "print('Test per-zone MAE (mg/dL):')",
        "print(pz_test_mae)",
    ),
    md(
        "## 8. Conclusions placeholder (write after the run)",
        "",
        "Compose the §8.3 conclusions paragraph in `reports/report.md` using the numbers printed above. Address explicitly:",
        "",
        "1. Pooled-MAE comparison vs GBM-300 at each horizon.",
        "2. Hypo-zone MAE comparison vs Persistence (9.06 / 18.34 / 27.26 at 30 / 60 / 90 min) and vs GBM-300 (10.42 / 26.75 / 40.72).",
        "3. LSTM vs GRU: any meaningful difference?",
        "4. Computational cost (epochs to converge × seconds per epoch on the chosen device).",
        "5. Whether the Phase C.1 result *strengthens* or *weakens* the case for C.2's asymmetric/zone-weighted loss intervention.",
        "",
        "**Next:** `notebooks/04c2_phase_c2_asymmetric_loss.ipynb` — same architecture, replace `MultiHorizonMSE` with `ZoneWeightedMSE` (over-weight the hypo zone) and `AsymmetricMSE` (over-penalise under-prediction); evaluate on the same test split.",
    ),
]


NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "notebooks" / "04c1_phase_c1_lstm_gru.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  cells={len(CELLS)}  size={out.stat().st_size} bytes")


if __name__ == "__main__":
    main()

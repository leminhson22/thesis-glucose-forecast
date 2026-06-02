# Thesis Glucose Forecast

Core reproducibility repository for a Type 1 Diabetes short-term glucose
forecasting thesis on the HUPA-UCM dataset.

The proposed model is a CNN-GRU-Attention network with
Persistence-Residual Learning. It predicts glucose at 30, 60, and 90 minutes
ahead from:

- dynamic 120-minute windows: `X_dynamic` with shape `(N, 24, 17)`;
- patient-level static features: `X_static` with shape `(N, 16)`;
- an appended patient index used only by the persistence-residual wrapper.

This repository intentionally keeps only the core files needed to reproduce
the training and result tables in Google Colab. It does not include thesis
drafts, full raw data dumps, generated Word documents, or non-core figures.

## Included core files

- `src/preprocessing.py`, `src/data_loading.py`, `src/config.py`: data
  cleaning, feature engineering, normalization, split, and sequence building.
- `src/baselines.py`: Persistence, Ridge, Random Forest, and GBM baselines.
- `src/models.py`: LSTM/GRU baselines and the proposed Hybrid CNN-GRU model.
- `src/run_phase_*.py`: baseline training/evaluation entry points.
- `src/run_step6_v2.py`: proposed model and ablation entry point.
- `src/evaluate.py`, `src/losses.py`, `src/train.py`, `src/datasets.py`:
  shared training and evaluation utilities.
- `data/processed/hupa_5min_sequences.npz`: train-ready sequence bundle.
- `outputs/models/scalers.json`: fitted normalization parameters.
- `outputs/models/step6_hybrid_v2_pers_resid.pt`: selected proposed-model
  checkpoint.
- `outputs/tables/*.csv`: saved comparison and main-model result tables.

## Google Colab quick start

Open `notebooks/00_colab_quickstart.ipynb` in Colab, or run:

```bash
git clone https://github.com/leminhson22/thesis-glucose-forecast.git
cd thesis-glucose-forecast
pip install -r requirements.txt
python src/run_phase_a.py --debug
python src/run_phase_c1.py --debug --model gru --epochs 1
python src/run_step6_v2.py --variant pers_resid --epochs 3
```

For full reproduction of the proposed model result, use:

```bash
python src/run_step6_v2.py --variant pers_resid --epochs 30
```

## Data contract

The train-ready file `data/processed/hupa_5min_sequences.npz` contains:

- `X_dynamic`: `(159172, 24, 17)`;
- `X_static`: `(159172, 16)`;
- `y`: `(159172, 3)` in mg/dL;
- `participant_ids`, `split`, `anchor_time`;
- dynamic/static feature-name arrays.

The chronological split is embedded in the `split` array.

## Clinical boundary

This repository is for research reproduction only. The model is not a medical
device and must not be used for insulin dosing, carbohydrate intake decisions,
or clinical treatment decisions.

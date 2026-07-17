# Thesis Glucose Forecast

Core reproducibility repository for a Type 1 Diabetes short-term glucose
forecasting thesis on the HUPA-UCM dataset.

The proposed model is a CNN-GRU-Attention network with
Persistence-Residual Learning. It predicts glucose at 30, 60, and 90 minutes
ahead from:

- dynamic 120-minute windows: `X_dynamic` with shape `(N, 24, 17)`;
- patient-level static features: `X_static` with shape `(N, 16)`;
- an appended patient index used only by the persistence-residual wrapper.

This repository intentionally keeps only the interview-ready thesis artefacts:
the six-model experimental pipeline, the processed HUPA-UCM bundle, selected
checkpoints, result tables, and the Streamlit interface. It does not include
thesis drafts, full raw data dumps, temporary verification folders, or
non-core experimental branches.

## Model scope

The public repository is scoped to the models presented in the thesis demo:

- Persistence
- Ridge Regression
- Random Forest
- LSTM
- GRU
- Hybrid CNN-GRU-Attention with Persistence-Residual Learning (proposed)

## Included core files

- `src/preprocessing.py`, `src/data_loading.py`, `src/config.py`: data
  cleaning, feature engineering, normalization, split, and sequence building.
- `src/baselines.py`: Persistence, Ridge, and Random Forest baselines.
- `src/models.py`: LSTM/GRU baselines and the proposed Hybrid CNN-GRU model.
- `src/run_phase_*.py`: baseline training/evaluation entry points.
- `src/run_step6_v2.py`: proposed model training/evaluation entry point.
- `src/evaluate.py`, `src/losses.py`, `src/train.py`, `src/datasets.py`:
  shared training and evaluation utilities.
- `data/processed/hupa_5min_sequences.npz`: train-ready sequence bundle.
- `outputs/models/scalers.json`: fitted normalization parameters.
- `outputs/models/ridge_phase_a.joblib`, `rf_phase_b.joblib`,
  `lstm_phase_c1.pt`, `gru_phase_c1.pt`: selected baseline checkpoints.
- `outputs/models/step6_hybrid_v2_pers_resid.pt`: selected proposed-model
  checkpoint.
- `outputs/tables/`: selected EDA, preprocessing, model-comparison, UQ, and
  XAI result tables for the six-model thesis narrative.
- `app.py`, `app/streamlit_app.py`, `.streamlit/config.toml`: Streamlit
  dashboard for local or Hugging Face Spaces demonstration.

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

## Streamlit dashboard

Run the local dashboard with:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The dashboard is a research-only demonstration. It loads the selected proposed
model checkpoint, precomputed Mondrian-ACI prediction intervals, and Integrated
Gradients explanation artefacts from `outputs/`.

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

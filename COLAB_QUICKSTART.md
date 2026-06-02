# Google Colab Quick Start

Use this file if you do not want to open the notebook.

## 1. Clone and install

```python
!git clone https://github.com/leminhson22/thesis-glucose-forecast.git
%cd thesis-glucose-forecast
!pip -q install -r requirements.txt
```

## 2. Check the train-ready data

```python
import numpy as np

d = np.load("data/processed/hupa_5min_sequences.npz", allow_pickle=True)
print("X_dynamic:", d["X_dynamic"].shape)
print("X_static:", d["X_static"].shape)
print("y:", d["y"].shape)
print("dynamic features:", [str(x) for x in d["feature_names_dynamic"]])
print("static features:", [str(x) for x in d["feature_names_static"]])
```

## 3. Smoke-test baselines

```python
!python src/run_phase_a.py --debug
!python src/run_phase_c1.py --debug --model gru --epochs 1
```

## 4. Train the proposed model

Fast Colab smoke test:

```python
!python src/run_step6_v2.py --variant pers_resid --epochs 3
```

Full reproduction:

```python
!python src/run_step6_v2.py --variant pers_resid --epochs 30
```

## 5. Inspect result tables

```python
import pandas as pd

pd.read_csv("outputs/tables/step6_v2_pers_resid_summary.csv")
```

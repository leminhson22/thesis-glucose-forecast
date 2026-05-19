"""HUPA-UCM data loading utilities.

All loaders are Colab-compatible: they accept a `base_path` argument that the
caller supplies via the standard `IN_COLAB` detection cell (see CLAUDE.md).
None of the functions assume a fixed local path.

Schema (per patient .xlsx file under data/data_hupa/Preprocessed/):
    time, glucose, calories, heart_rate, steps,
    basal_rate, bolus_volume_delivered, carb_input

Static metadata (data/data_hupa/patient_data_characteristic.xlsx):
    Patient ID, Gender, HbAc [%], Age [years], DX Time [years],
    Weight [kg], Height [cm], Treatment (CSII/MDI)

Both groups are joined by participant_id (e.g., "HUPA0001P").
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_SUBDIR = os.path.join("data", "data_hupa")
PREPROCESSED_SUBDIR = os.path.join(DATA_SUBDIR, "Preprocessed")
CHARACTERISTIC_FILE = "patient_data_characteristic.xlsx"

# Canonical column order for the per-patient timeseries dataframe.
TS_COLUMNS = [
    "time",
    "glucose",
    "calories",
    "heart_rate",
    "steps",
    "basal_rate",
    "bolus_volume_delivered",
    "carb_input",
]

# Mapping from the original characteristic columns to snake_case names
# that join cleanly with downstream feature tables.
STATIC_COLUMN_RENAME = {
    "Patient ID": "participant_id",
    "Gender": "gender",
    "HbAc [%]": "hba1c_pct",
    "Age [years]": "age_years",
    "DX Time [years]": "dx_time_years",
    "Weight [kg]": "weight_kg",
    "Height [cm]": "height_cm",
    "Treatment (CSII/MDI)": "treatment",
}


def _resolve_base(base_path: str | os.PathLike | None) -> Path:
    """Return an absolute Path, falling back to the project root if unset.

    The fallback assumes the script lives in <project_root>/src/.
    """
    if base_path is not None:
        return Path(base_path)
    return Path(__file__).resolve().parent.parent


def load_hupa_patient(
    participant_id: str,
    base_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Load one HUPA participant's pre-processed timeseries.

    Parameters
    ----------
    participant_id
        E.g. ``"HUPA0001P"``. Pass exactly as it appears in the filename.
    base_path
        Project root (typically the ``BASE_PATH`` value computed by the
        Colab detection cell). If ``None``, falls back to two levels above
        this file.

    Returns
    -------
    DataFrame with the ``TS_COLUMNS`` schema, sorted by ``time`` ascending,
    plus a ``participant_id`` column.
    """
    base = _resolve_base(base_path)
    fpath = base / PREPROCESSED_SUBDIR / f"{participant_id}.xlsx"
    if not fpath.exists():
        raise FileNotFoundError(f"HUPA file not found: {fpath}")
    df = pd.read_excel(fpath)
    # The pre-processed files already have the canonical schema. Sort
    # defensively in case a future release does not.
    df = df.sort_values("time").reset_index(drop=True)
    df.insert(0, "participant_id", participant_id)
    return df


def list_hupa_participants(base_path: str | os.PathLike | None = None) -> list[str]:
    """Return sorted list of all 25 participant IDs available locally."""
    base = _resolve_base(base_path)
    folder = base / PREPROCESSED_SUBDIR
    if not folder.exists():
        raise FileNotFoundError(f"Preprocessed folder not found: {folder}")
    ids = []
    for f in folder.iterdir():
        # Skip the .csv mirror copies that ship alongside the .xlsx files.
        if f.suffix.lower() == ".xlsx" and f.name.startswith("HUPA"):
            ids.append(f.stem)
    return sorted(ids)


def load_hupa_all(base_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Concatenate every participant's timeseries into one long DataFrame.

    Result has shape (309_392, 9) on the released cohort. Add the static
    metadata via :func:`attach_static_metadata` if you need clinical
    covariates joined.
    """
    parts = [load_hupa_patient(pid, base_path) for pid in list_hupa_participants(base_path)]
    return pd.concat(parts, axis=0, ignore_index=True)


def load_patient_characteristics(
    base_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Load the per-patient clinical static metadata table.

    Returns a DataFrame with columns:
        participant_id, gender, hba1c_pct, age_years, dx_time_years,
        weight_kg, height_cm, treatment, bmi

    BMI is computed from ``weight_kg / (height_cm / 100) ** 2``.
    """
    base = _resolve_base(base_path)
    fpath = base / DATA_SUBDIR / CHARACTERISTIC_FILE
    if not fpath.exists():
        raise FileNotFoundError(f"Characteristic file not found: {fpath}")
    df = pd.read_excel(fpath)
    df = df.rename(columns=STATIC_COLUMN_RENAME)
    df["participant_id"] = df["participant_id"].str.strip()
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
    return df


def attach_static_metadata(
    ts_df: pd.DataFrame,
    base_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Left-join the static clinical metadata onto a per-timestep dataframe."""
    static = load_patient_characteristics(base_path)
    return ts_df.merge(static, on="participant_id", how="left")


def summarise_cohort(base_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """One-row-per-patient summary used by notebooks/01_data_understanding.

    Columns produced:
        participant_id, n_rows, days, time_start, time_end,
        glucose_mean, glucose_std, glucose_min, glucose_max,
        pct_hypo, pct_in_range, pct_hyper,
        pct_glucose_low_cap, pct_glucose_high_extreme,
        basal_recording_pct, bolus_events, carb_events,
        hba1c_pct, age_years, treatment, bmi
    """
    rows = []
    static = load_patient_characteristics(base_path).set_index("participant_id")
    for pid in list_hupa_participants(base_path):
        df = load_hupa_patient(pid, base_path)
        n = len(df)
        g = df["glucose"]
        row = {
            "participant_id": pid,
            "n_rows": n,
            "days": round(n * 5 / 1440, 1),
            "time_start": df["time"].iloc[0],
            "time_end": df["time"].iloc[-1],
            "glucose_mean": g.mean(),
            "glucose_std": g.std(),
            "glucose_min": g.min(),
            "glucose_max": g.max(),
            "pct_hypo": 100 * (g < 70).mean(),
            "pct_in_range": 100 * ((g >= 70) & (g <= 180)).mean(),
            "pct_hyper": 100 * (g > 180).mean(),
            "pct_glucose_low_cap": 100 * (g == 40).mean(),
            "pct_glucose_high_extreme": 100 * (g > 400).mean(),
            "basal_recording_pct": 100 * (df["basal_rate"] > 0).mean(),
            "bolus_events": int((df["bolus_volume_delivered"] > 0).sum()),
            "carb_events": int((df["carb_input"] > 0).sum()),
        }
        if pid in static.index:
            s = static.loc[pid]
            row.update(
                {
                    "hba1c_pct": s["hba1c_pct"],
                    "age_years": s["age_years"],
                    "treatment": s["treatment"],
                    "bmi": s["bmi"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)

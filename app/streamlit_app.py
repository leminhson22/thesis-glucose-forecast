"""Streamlit MVP for the proposed glucose-forecasting model.

A research demo, not a clinical tool. Loads the proposed
CNN-GRU-Attention + Persistence-Residual checkpoint (§7.6 of the thesis
report) and visualises, for any held-out test window:

  * the most recent 2 hours of CGM glucose;
  * the model's 30 / 60 / 90-minute forecast;
  * a calibrated 90 % prediction interval from Section 9.8 Mondrian-ACI;
  * a hypoglycaemia-risk indicator;
  * an Integrated Gradients explanation heat-map (§10) showing which
    inputs at which lookback steps drove the forecast.

The app loads everything from pre-computed artefacts under outputs/; the
model itself is loaded once and reused for on-demand IG when the selected
window is not in the case-study parquet.

Run locally with:

    streamlit run app.py

This file deliberately is *not* a clinical-decision-support system. The
header banner in the UI states this and a research-only disclaimer is
shown alongside every forecast.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.request
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from data_loading import load_patient_characteristics  # noqa: E402
from datasets import load_npz_splits  # noqa: E402
from eval_step6_v2 import load_variant_model  # noqa: E402
from explain import integrated_gradients_dyn, temporal_feature_heatmap  # noqa: E402
from run_step6_v2 import attach_pid_index_to_static, load_pid_scaler_table  # noqa: E402
from evaluate import clarke_eg_zones  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
FIG_DIR = ROOT / "outputs" / "figures"

HORIZONS = list(C.HORIZON_MINUTES)
HORIZON_LABEL = {30: "30 min", 60: "60 min", 90: "90 min"}
ZONE_COLOUR = {"hypo": "#d62728", "tir": "#2ca02c", "hyper": "#ff7f0e"}
LOGO_PATH = ROOT / "app" / "assets" / "vnu_is_logo.png"
INTERVAL_TABLE = TABLES / "uq_aci_alpha_trajectory.parquet"
INTERVAL_LO_COL = "lower_aci"
INTERVAL_UP_COL = "upper_aci"
INTERVAL_ALPHA_COL = "alpha_t"
INTERVAL_LABEL = "90% Mondrian-ACI PI"
INTERVAL_SHORT_LABEL = "90% ACI PI"
INTERVAL_METHOD_LABEL = "Mondrian-ACI"

st.set_page_config(
    page_title="Glucose-Forecast Research Demo (HUPA-UCM)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --vnu-green: #123A6F;
        --vnu-green-dark: #0B2E5F;
        --vnu-gold: #D9A323;
        --vnu-gold-bright: #F0C44C;
        --app-bg: #FFFFFF;
        --app-surface: #FFFFFF;
        --app-surface-soft: #F8FAFC;
        --app-ivory: #FFFFFF;
        --app-text: #102A43;
        --app-muted: #8E959C;
        --app-border: #D9E1D8;
        --shadow-soft: 0 10px 28px rgba(15, 46, 87, 0.12);
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--app-bg);
        color: var(--app-text);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 2px solid rgba(217, 163, 35, 0.45);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--vnu-green-dark);
        letter-spacing: 0;
    }
    .app-hero {
        background:
            linear-gradient(135deg, rgba(15, 46, 87, 0.96) 0%, rgba(18, 58, 111, 0.96) 55%, rgba(217, 163, 35, 0.94) 100%);
        border: 1px solid rgba(217, 163, 35, 0.48);
        border-top: 0;
        border-radius: 16px;
        padding: 1.18rem 1.35rem;
        box-shadow: 0 14px 34px rgba(15, 46, 87, 0.20);
        margin-bottom: 1rem;
    }
    .hero-row {
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
    }
    .hero-logo {
        width: 46px;
        height: 46px;
        object-fit: contain;
        border-radius: 8px;
        background: #FFFFFF;
        border: 1px solid var(--app-border);
        padding: 0.22rem;
        flex: 0 0 auto;
        image-rendering: auto;
    }
    .sidebar-logo-wrap {
        margin: -1.2rem 0 0.85rem 0;
        padding: 0.15rem 0 0.55rem 0;
        line-height: 1;
        border-bottom: 1px solid var(--app-border);
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .sidebar-logo {
        width: 58px;
        height: auto;
        display: block;
        object-fit: contain;
        image-rendering: auto;
        filter: contrast(1.08) saturate(1.08);
    }
    .sidebar-brand-text {
        color: var(--vnu-green-dark);
        font-size: 0.78rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .sidebar-brand-sub {
        color: var(--app-muted);
        font-size: 0.68rem;
        font-weight: 600;
    }
    .app-hero h1 {
        margin: 0;
        color: #FFF7E0;
        font-size: 1.78rem;
        line-height: 1.2;
        letter-spacing: 0;
    }
    .app-hero p {
        margin: 0.35rem 0 0;
        color: #F8F3E3;
        font-size: 0.98rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.28rem 0.68rem;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid rgba(255, 247, 224, 0.62);
        background: #F0C44C;
        color: #0B2E5F;
        white-space: nowrap;
        margin-top: 0.55rem;
    }
    .section-title {
        color: var(--vnu-green-dark);
        font-size: 1.05rem;
        font-weight: 750;
        margin: 0.45rem 0 0.55rem;
    }
    .control-block {
        background: #FFFFFF;
        border: 1px solid rgba(217, 163, 35, 0.28);
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        margin: 0.65rem 0;
    }
    .patient-card, .research-note {
        background: var(--app-surface);
        border: 1px solid var(--app-border);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 4px 12px rgba(16, 42, 36, 0.045);
    }
    .patient-card .label {
        color: var(--app-muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        font-weight: 700;
    }
    .patient-card .value {
        color: var(--app-text);
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }
    .panel-card {
        background: var(--app-surface);
        border: 1px solid var(--app-border);
        border-radius: 16px;
        padding: 1rem 1rem 0.75rem;
        box-shadow: var(--shadow-soft);
        border-top: 4px solid var(--vnu-gold);
        margin: 0.75rem 0 1rem;
    }
    .panel-card h3 {
        margin: 0 0 0.25rem;
        color: var(--vnu-green-dark);
        font-size: 1.05rem;
        letter-spacing: 0;
    }
    .panel-subtitle {
        margin: 0 0 0.75rem;
        color: var(--app-muted);
        font-size: 0.86rem;
    }
    .inline-figure {
        margin: 0.2rem 0 1rem;
    }
    .inline-figure img {
        display: block;
        width: 100%;
        height: auto;
        border: 1px solid var(--app-border);
        border-radius: 12px;
        background: #FFFFFF;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFFDF7 100%);
        border: 1px solid var(--app-border);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        box-shadow: var(--shadow-soft);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        color: var(--app-muted);
    }
    div[data-testid="stMetricValue"] {
        color: #0B2E5F;
        font-size: 1.42rem;
        font-weight: 800;
    }
    .risk-card {
        border: 1px solid var(--app-border);
        border-left: 5px solid #64748b;
        border-radius: 16px;
        padding: 0.86rem 0.92rem;
        background: #FFFFFF;
        margin-bottom: 0.65rem;
        box-shadow: 0 10px 24px rgba(15, 46, 87, 0.10);
        min-height: 142px;
        height: 142px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    .risk-card .risk-top {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        align-items: center;
        min-height: 28px;
        margin-bottom: 0.25rem;
    }
    .risk-card .horizon {
        font-weight: 800;
        color: var(--app-text);
        white-space: nowrap;
    }
    .risk-card .risk-badge {
        border-radius: 999px;
        padding: 0.17rem 0.46rem;
        font-size: 0.68rem;
        font-weight: 800;
        white-space: nowrap;
        text-align: center;
        flex: 0 0 auto;
    }
    .risk-card .risk-values {
        color: var(--app-muted);
        font-size: 0.83rem;
        line-height: 1.45;
        min-height: 46px;
        overflow-wrap: anywhere;
    }
    .threshold-chip {
        display: inline-flex;
        margin-top: 0.45rem;
        border-radius: 999px;
        padding: 0.16rem 0.48rem;
        background: #FFF4C7;
        color: #0F2E57;
        font-size: 0.68rem;
        font-weight: 800;
        border: 1px solid #F3D45D;
        min-height: 22px;
        align-items: center;
        width: fit-content;
        max-width: 100%;
        white-space: nowrap;
    }
    .threshold-chip.placeholder {
        visibility: hidden;
    }
    .risk-card.hypo {border-left-color: #C62828; background: #FFF1F2;}
    .risk-card.hypo .risk-badge {background: #FFD6DA; color: #8A1111;}
    .risk-card.hyper {border-left-color: #D9A323; background: #FFF4D6;}
    .risk-card.hyper .risk-badge {background: #F0C44C; color: #0F2E57;}
    .risk-card.tir {border-left-color: #123A6F; background: #F8FDF9;}
    .risk-card.tir .risk-badge {background: #DDFBEA; color: #0F5F3D;}
    .clarke-card {
        border: 1px solid var(--app-border);
        border-left: 5px solid #64748b;
        border-radius: 14px;
        padding: 0.78rem 0.86rem;
        background: #FFFFFF;
        min-height: 128px;
        box-shadow: 0 8px 20px rgba(15, 46, 87, 0.08);
    }
    .clarke-card .zone {
        font-size: 1.32rem;
        line-height: 1.1;
        font-weight: 850;
        margin-bottom: 0.22rem;
    }
    .clarke-card .horizon {
        color: var(--app-muted);
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
    }
    .clarke-card .desc {
        color: var(--app-text);
        font-size: 0.82rem;
        line-height: 1.38;
        margin-top: 0.36rem;
    }
    .clarke-card.zone-a {border-left-color: #15803d; background: #F0FDF4;}
    .clarke-card.zone-a .zone {color: #166534;}
    .clarke-card.zone-b {border-left-color: #0F2E57; background: #EFF6FF;}
    .clarke-card.zone-b .zone {color: #0F2E57;}
    .clarke-card.zone-c {border-left-color: #B45309; background: #FFFBEB;}
    .clarke-card.zone-c .zone {color: #92400E;}
    .clarke-card.zone-d, .clarke-card.zone-e {border-left-color: #B91C1C; background: #FEF2F2;}
    .clarke-card.zone-d .zone, .clarke-card.zone-e .zone {color: #991B1B;}
    .xai-note {
        background: #FFF7E0;
        border: 1px solid var(--app-border);
        border-left: 5px solid var(--vnu-gold);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.2rem 0 0.9rem 0;
        color: var(--app-text);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .small-muted {
        color: var(--app-muted);
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .stDataFrame {
        border: 1px solid var(--app-border);
        border-radius: 10px;
        overflow: hidden;
    }
    .metric-risk div[data-testid="stMetric"] {
        border-left: 6px solid var(--vnu-gold);
        background: linear-gradient(180deg, #FFF9EC 0%, #FFF4D6 100%);
    }
    .metric-navy div[data-testid="stMetric"] {
        border-left: 6px solid var(--vnu-green);
        background: linear-gradient(180deg, #FFFFFF 0%, #EEF3FA 100%);
    }
    .metric-uncertainty div[data-testid="stMetric"] {
        border-left: 6px solid #6F7E8F;
        background: linear-gradient(180deg, #FFFFFF 0%, #F3F5F6 100%);
    }
    .metric-neutral div[data-testid="stMetric"] {
        border-left: 6px solid #0F2E57;
        background: #FFFFFF;
    }
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid var(--app-border);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 18px rgba(15, 46, 87, 0.08);
        font-size: 0.86rem;
        margin-top: 0.7rem;
    }
    .styled-table thead th {
        background: #0F2E57;
        color: #FFF7E0;
        font-weight: 800;
        padding: 0.62rem 0.7rem;
        text-align: left;
        white-space: nowrap;
    }
    .styled-table tbody td {
        padding: 0.56rem 0.7rem;
        border-top: 1px solid #E7DDD0;
        color: #102A43;
    }
    .styled-table tbody tr:nth-child(even) td {
        background: #FFFFFF;
    }
    .styled-table tbody tr:nth-child(odd) td {
        background: #FFFFFF;
    }
    .styled-table .emph {
        color: #0F2E57;
        font-weight: 800;
    }
    .footer-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        border: 1px solid var(--app-border);
        background: #FFFFFF;
        color: var(--app-text);
        border-radius: 999px;
        padding: 0.38rem 0.65rem;
        margin: 0.12rem 0.18rem 0.12rem 0;
        font-size: 0.8rem;
        box-shadow: 0 3px 10px rgba(16, 42, 36, 0.045);
    }
    button[kind="primary"], .stButton button {
        border-radius: 8px !important;
        border: 1px solid var(--vnu-green) !important;
    }
    @media (max-width: 900px) {
        .app-hero h1 {font-size: 1.35rem;}
        .panel-card {padding: 0.85rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

def load_aci_intervals() -> pd.DataFrame:
    intervals = pd.read_parquet(INTERVAL_TABLE)
    required = {
        "participant_id",
        "sample_idx",
        "horizon_min",
        "y_true",
        "y_pred",
        "zone",
        INTERVAL_ALPHA_COL,
        INTERVAL_LO_COL,
        INTERVAL_UP_COL,
    }
    missing = required.difference(intervals.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{INTERVAL_TABLE} is missing required columns: {missing_cols}")

    intervals = intervals.copy()
    intervals["abs_err"] = (intervals["y_true"] - intervals["y_pred"]).abs()
    intervals["pi_width"] = intervals[INTERVAL_UP_COL] - intervals[INTERVAL_LO_COL]
    return intervals


@st.cache_resource(show_spinner="Loading test split + checkpoints …")
def load_everything():
    splits = load_npz_splits(ROOT / C.SEQUENCES_NPZ)
    _, _, pid_lookup = load_pid_scaler_table(splits)
    work = attach_pid_index_to_static(splits, pid_lookup)

    model = load_variant_model(
        "pers_resid",
        MODELS / "step6_hybrid_v2_pers_resid.pt",
        n_dynamic=len(splits["feat_dyn"]),
        n_static_dataset=work["train"]["X_static"].shape[1],
        feat_dyn=splits["feat_dyn"],
        splits=splits,
    )
    model.eval()

    intervals = load_aci_intervals()
    cases = pd.read_parquet(TABLES / "xai_ig_case_studies.parquet")
    static_csv = pd.read_csv(ROOT / "data" / "processed" / "hupa_static_features.csv")
    clinical_df = load_patient_characteristics(ROOT)
    # Keep the UI consistent with the modelling pipeline override.
    clinical_df.loc[clinical_df["participant_id"] == "HUPA0011P", "treatment"] = "MDI"
    global_imp = pd.read_csv(TABLES / "xai_ig_global_importance.csv")

    return {
        "splits": splits,
        "work": work,
        "model": model,
        "intervals": intervals,
        "cases": cases,
        "static_df": static_csv,
        "clinical_df": clinical_df,
        "global_imp": global_imp,
        "feat_dyn": splits["feat_dyn"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zone_of_value(v: float) -> str:
    if v < 70:
        return "hypo"
    if v > 180:
        return "hyper"
    return "tir"


def logo_img_html() -> str:
    if not LOGO_PATH.exists():
        return ""
    data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f'<img class="hero-logo" src="data:image/png;base64,{data}" alt="VNU-IS logo">'


def sidebar_logo_html() -> str:
    if not LOGO_PATH.exists():
        return ""
    data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return (
        f'<div class="sidebar-logo-wrap">'
        f'<img class="sidebar-logo" src="data:image/png;base64,{data}" alt="VNU-IS logo">'
        f'<div><div class="sidebar-brand-text">VNU-IS</div>'
        f'<div class="sidebar-brand-sub">Medical AI Research</div></div>'
        f'</div>'
    )


def fig_to_inline_html(fig, alt: str) -> str:
    """Render a Matplotlib figure as inline base64 HTML.

    Hugging Face Spaces can occasionally lose the Streamlit media server URI
    during websocket reconnects. Inline images avoid st.pyplot media URLs.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<div class="inline-figure"><img src="data:image/png;base64,{data}" alt="{alt}"></div>'


def fmt_zone(zone: str) -> str:
    return {
        "hypo": "Hypoglycaemia (<70 mg/dL)",
        "tir": "Time-in-range (70–180 mg/dL)",
        "hyper": "Hyperglycaemia (>180 mg/dL)",
    }[zone]


def risk_class(y_pred: float, lower: float, upper: float) -> str:
    if y_pred < 70 or lower < 70:
        return "hypo"
    if y_pred > 180 or upper > 180:
        return "hyper"
    return "tir"


def risk_label(zone: str) -> str:
    return {
        "hypo": "Hypoglycaemia risk",
        "tir": "Time-in-range",
        "hyper": "Hyperglycaemia risk",
    }[zone]


def clarke_zone_label(y_true: float, y_pred: float) -> str:
    """Return Clarke Error Grid zone A-E for one forecast/reference pair."""
    zone = clarke_eg_zones(np.asarray([y_true], dtype=float), np.asarray([y_pred], dtype=float))[0]
    return str(zone)


def clarke_zone_description(zone: str) -> str:
    return {
        "A": "Clinically accurate: prediction is close enough to the reference value.",
        "B": "Benign error: deviation is usually not expected to lead to inappropriate treatment.",
        "C": "Over-correction risk: error may suggest unnecessary corrective action.",
        "D": "Failure-to-detect risk: dangerous hypo/hyper state may be missed.",
        "E": "Erroneous-treatment risk: prediction points to the opposite clinical condition.",
    }.get(zone, "Unclassified Clarke zone.")


def clarke_zone_severity(zone: str) -> str:
    if zone == "A":
        return "Accurate"
    if zone == "B":
        return "Benign"
    return "Clinically risky"


def render_clarke_table(rows: list[dict]) -> str:
    columns = [
        ("Horizon", "Horizon"),
        ("Observed", "Observed"),
        ("Forecast", "Forecast"),
        ("Clarke zone", "Clarke zone"),
        ("Interpretation", "Interpretation"),
        ("Abs. error", "Abs. error"),
    ]
    html = ['<table class="styled-table"><thead><tr>']
    html.extend(f"<th>{label}</th>" for label, _ in columns)
    html.append("</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>")
        for label, key in columns:
            cls = ' class="emph"' if label in {"Clarke zone", "Abs. error"} else ""
            html.append(f"<td{cls}>{row.get(key, '')}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def select_case_sample(pid: str, pid_sample_idx: np.ndarray, intervals_df: pd.DataFrame, case_name: str) -> int | None:
    """Pick a representative held-out sample for a demo scenario."""
    if case_name == "Manual slider":
        return None

    h30, h60, h90 = HORIZONS
    sub = intervals_df[
        (intervals_df["participant_id"] == pid)
        & (intervals_df["sample_idx"].isin(pid_sample_idx))
    ]
    if sub.empty:
        return None

    pred = sub.pivot(index="sample_idx", columns="horizon_min", values="y_pred")
    err = sub.pivot(index="sample_idx", columns="horizon_min", values="abs_err")
    lo = sub.pivot(index="sample_idx", columns="horizon_min", values=INTERVAL_LO_COL)
    up = sub.pivot(index="sample_idx", columns="horizon_min", values=INTERVAL_UP_COL)
    needed = [h30, h60, h90]
    pred = pred.dropna(subset=needed)
    if pred.empty:
        return None

    if case_name == "Hypoglycaemia risk":
        score = np.minimum(pred.min(axis=1), lo.reindex(pred.index).min(axis=1))
        return int(score.idxmin())
    if case_name == "Hyperglycaemia risk":
        score = np.maximum(pred.max(axis=1), up.reindex(pred.index).max(axis=1))
        return int(score.idxmax())
    if case_name == "Rising glucose":
        return int((pred[h90] - pred[h30]).idxmax())
    if case_name == "Falling glucose":
        return int((pred[h90] - pred[h30]).idxmin())
    if case_name == "Large model error":
        return int(err.reindex(pred.index).mean(axis=1).idxmax())
    if case_name == "Wide uncertainty interval":
        width = up.reindex(pred.index) - lo.reindex(pred.index)
        return int(width.mean(axis=1).idxmax())
    if case_name == "Stable glucose":
        pred_range = pred.max(axis=1) - pred.min(axis=1)
        in_range = ((pred >= 70) & (pred <= 180)).all(axis=1)
        candidates = pred_range[in_range]
        return int(candidates.idxmin() if not candidates.empty else pred_range.idxmin())

    return None


def feature_label(feature: str) -> str:
    labels = {
        "glucose": "recent glucose level",
        "glucose_velocity": "recent glucose rate of change",
        "glucose_30m_mean": "30-minute glucose average",
        "glucose_60m_mean": "60-minute glucose average",
        "glucose_120m_mean": "120-minute glucose average",
        "glucose_60m_std": "recent glucose variability",
        "heart_rate": "heart rate",
        "heart_rate_30m_mean": "30-minute heart-rate average",
        "basal_rate": "basal insulin rate",
        "basal_coverage_24h": "basal-data coverage",
        "bolus_60m_sum": "recent bolus insulin",
        "insulin_on_board": "estimated insulin on board",
        "carbs_on_board": "estimated carbohydrates on board",
        "steps_150m_sum": "recent step count",
        "hour_sin": "time of day",
        "hour_cos": "time of day",
        "glucose_low_cap": "sensor low-cap flag",
    }
    return labels.get(feature, feature.replace("_", " "))


def explanation_text(horizon: int, y_pred: float, lo: float | None, up: float | None, rank_df: pd.DataFrame) -> str:
    top = rank_df.head(3)
    feature_phrase = ", ".join(
        f"{feature_label(row.feature)} ({row.share_pct:.1f}%)"
        for row in top.itertuples(index=False)
    )
    lower = lo if lo is not None else y_pred
    upper = up if up is not None else y_pred
    zone = risk_class(y_pred, lower, upper)
    interval_phrase = ""
    if lo is not None and up is not None:
        interval_phrase = f" The {INTERVAL_LABEL} is [{lo:.0f}, {up:.0f}] mg/dL."
    return (
        f"For the +{horizon} minute forecast, the model prediction is {y_pred:.0f} mg/dL "
        f"and the displayed risk category is {risk_label(zone).lower()}.{interval_phrase} "
        f"Integrated Gradients assigns the largest attribution mass to {feature_phrase}. "
        "This is an attribution summary for model inspection only, not a treatment recommendation."
    )


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a config value from Streamlit secrets first, then environment."""
    if name == "OPENAI_API_KEY":
        session_key = st.session_state.get("openai_api_key")
        if session_key:
            return str(session_key)

    value = None
    secret_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        ROOT / ".streamlit" / "secrets.toml",
    ]
    if any(path.exists() for path in secret_paths):
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
    if value:
        return str(value)
    return os.environ.get(name, default)


def build_llm_prompt(
    pid: str,
    horizon: int,
    last_glucose: float,
    y_pred: float,
    lo: float | None,
    up: float | None,
    rank_df: pd.DataFrame,
    patient_row: pd.Series | None,
) -> str:
    top_features = [
        {"feature": str(row.feature), "plain_meaning": feature_label(str(row.feature)), "share_pct": float(row.share_pct)}
        for row in rank_df.head(6).itertuples(index=False)
    ]
    patient_context = {}
    if patient_row is not None:
        patient_context = {
            "gender": patient_row.get("gender"),
            "treatment": patient_row.get("treatment"),
            "age_years": float(patient_row.get("age_years")) if pd.notna(patient_row.get("age_years")) else None,
            "hba1c_pct": float(patient_row.get("hba1c_pct")) if pd.notna(patient_row.get("hba1c_pct")) else None,
            "bmi": float(patient_row.get("bmi")) if pd.notna(patient_row.get("bmi")) else None,
        }

    payload = {
        "participant_id": pid,
        "current_glucose_mgdl": round(float(last_glucose), 1),
        "horizon_min": int(horizon),
        "forecast_mgdl": round(float(y_pred), 1),
        "mondrian_aci_prediction_interval_90_mgdl": None if lo is None or up is None else [round(float(lo), 1), round(float(up), 1)],
        "risk_category": risk_label(risk_class(float(y_pred), lo if lo is not None else float(y_pred), up if up is not None else float(y_pred))),
        "top_integrated_gradients_features": top_features,
        "patient_context": patient_context,
    }
    return (
        "You are writing a short plain-language explanation for a research dashboard about "
        "short-term glucose forecasting in type 1 diabetes. Use only the JSON data below. "
        "Do not provide medical advice, dosing advice, carbohydrate recommendations, or instructions "
        "for patient action. Do not say the model is clinically safe. Explain what the model predicted, "
        "what the uncertainty interval means, and which input signals the Integrated Gradients attribution "
        "suggests were influential. End with one sentence saying this is a research explanation only and "
        "not a substitute for clinical judgement. Write 4-6 concise sentences in Vietnamese.\n\n"
        f"JSON data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def call_llm_explanation(prompt: str) -> tuple[str | None, str | None]:
    """Call an OpenAI-compatible chat-completions endpoint if configured."""
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is not configured."

    base_url = get_secret("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = get_secret("OPENAI_MODEL", "gpt-4.1-mini")
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 280,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain model outputs for a research prototype. "
                    "You must refuse to provide medical or treatment advice."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip(), None
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return None, f"LLM request failed: {exc}"


def call_ollama_explanation(prompt: str, model_name: str = "llama3.2") -> tuple[str | None, str | None]:
    """Call a local Ollama chat endpoint. No paid API is used."""
    url = "http://localhost:11434/api/chat"
    body = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain model outputs for a research prototype. "
                    "You must refuse to provide medical or treatment advice."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2, "num_predict": 260},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"].strip(), None
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
        return None, f"Ollama request failed: {exc}"


def render_forecast_table(rows: list[dict]) -> str:
    columns = [
        ("Horizon", "Horizon"),
        ("Forecast", "Forecast (mg/dL)"),
        (INTERVAL_SHORT_LABEL, "90 % PI lower"),
        ("Width", "PI width"),
        ("Observed", "Observed (held-out)"),
        ("Model AE", "Model abs error"),
    ]
    html = ['<table class="styled-table"><thead><tr>']
    html.extend(f"<th>{label}</th>" for label, _ in columns)
    html.append("</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>")
        for label, key in columns:
            cls = ' class="emph"' if label in {"Forecast", "Width", "Model AE"} else ""
            if label == INTERVAL_SHORT_LABEL:
                cell = f"[{row.get('90 % PI lower', '')}, {row.get('90 % PI upper', '')}]"
            else:
                cell = row.get(key, "")
            html.append(f"<td{cls}>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def compute_ig_for_window(model, x_dyn_np, x_stat_np, horizon_idx) -> np.ndarray:
    xd = torch.from_numpy(x_dyn_np[None]).float()
    xs = torch.from_numpy(x_stat_np[None]).float()
    a = integrated_gradients_dyn(model, xd, xs, horizon_idx=horizon_idx, m=50)
    return a[0].numpy()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

state = load_everything()
splits = state["splits"]
work = state["work"]
model = state["model"]
intervals = state["intervals"]
static_df = state["static_df"]
clinical_df = state["clinical_df"]
feat_dyn = state["feat_dyn"]
test = work["test"]
n_samples = test["X_dynamic"].shape[0]

st.markdown(
    """
    <div class="app-hero">
        <h1>Short-Term Glucose Forecasting Dashboard</h1>
        <p>
            CNN-GRU-Attention with persistence-residual learning, Mondrian-ACI adaptive
            uncertainty, Clarke Error Grid safety review, and Integrated Gradients attribution on the held-out HUPA-UCM test split.
        </p>
        <span class="badge">Research demo · Not medical advice</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if False:
    pass
    # Legacy Streamlit title/warning disabled after custom hero redesign.
    # st.caption(
    "Proposed model — CNN-GRU-Attention with Persistence-Residual Learning (§7.6) — "
    "wrapped with Mondrian-ACI prediction intervals (Section 9.8) and Integrated-Gradients "
    "feature attributions (§10). Held-out HUPA-UCM test split."
    # )
    # st.warning(
    "**Research demo only.** Forecasts are not medical advice and must not replace "
    "clinician judgement.",
    # icon="warning",
    # )

# Sidebar — sample picker
with st.sidebar:
    if LOGO_PATH.exists():
        st.markdown(sidebar_logo_html(), unsafe_allow_html=True)
    else:
        st.caption("Logo: save VNU-IS logo to `app/assets/vnu_is_logo.png`.")
    st.markdown('<div class="section-title">Test Window Selection</div>', unsafe_allow_html=True)

    all_pids = sorted(np.unique(test["pids"]).tolist())
    pid = st.selectbox("Patient", all_pids, index=min(13, len(all_pids) - 1))

    pid_mask = test["pids"] == pid
    pid_sample_idx = np.where(pid_mask)[0]
    if len(pid_sample_idx) == 0:
        st.error(f"No test windows for {pid}")
        st.stop()

    case_name = st.selectbox(
        "Demo scenario",
        [
            "Manual slider",
            "Stable glucose",
            "Rising glucose",
            "Falling glucose",
            "Hypoglycaemia risk",
            "Hyperglycaemia risk",
            "Large model error",
            "Wide uncertainty interval",
        ],
        index=0,
        help="Use curated scenarios for faster demos, or choose Manual slider to inspect any test window.",
    )
    selected_case_idx = select_case_sample(pid, pid_sample_idx, intervals, case_name)

    if selected_case_idx is None:
        pos = st.slider(
            "Window index within this patient",
            min_value=0,
            max_value=len(pid_sample_idx) - 1,
            value=min(120, len(pid_sample_idx) - 1),
            step=1,
            help="Each step is one 5-minute forecast tick within this patient's test split.",
        )
        sample_idx = int(pid_sample_idx[pos])
    else:
        sample_idx = int(selected_case_idx)
        pos = int(np.where(pid_sample_idx == sample_idx)[0][0])
        st.caption(f"Selected representative window: index {pos} within this patient's test split.")

    st.markdown('<div class="small-muted">Explanation options</div>', unsafe_allow_html=True)
    explanation_mode = st.selectbox(
        "Explanation mode",
        ["Deterministic", "Ollama local", "OpenAI API"],
        index=0,
        help="Ollama runs locally without paid API calls. OpenAI API requires a valid key.",
    )
    use_llm = explanation_mode != "Deterministic"
    ollama_model = "llama3.2"
    if explanation_mode == "Ollama local":
        ollama_model = st.text_input(
            "Ollama model",
            value="llama3.2",
            help="Run `ollama pull llama3.2` first, then keep Ollama running locally.",
        )
        st.caption("Local endpoint: http://localhost:11434")
    elif explanation_mode == "OpenAI API":
        st.text_input(
            "OpenAI API key",
            type="password",
            key="openai_api_key",
            help="Optional session-only key. It is not written to disk by this app.",
            placeholder="sk-...",
        )
        if not get_secret("OPENAI_API_KEY"):
            st.caption("OpenAI explanation disabled until an API key is entered or configured.")

    st.markdown("---")
    st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)
    clinical_row = clinical_df[clinical_df["participant_id"] == pid]
    feature_row = static_df[static_df["participant_id"] == pid]
    if not clinical_row.empty:
        r = clinical_row.iloc[0].to_dict()
        gender_txt = r.get("gender", "n/a")
        treatment_txt = r.get("treatment", "n/a")
        age_txt = f"{r['age_years']:.0f} years" if pd.notna(r.get("age_years")) else "n/a"
        hba1c_txt = f"{r['hba1c_pct']:.1f}%" if pd.notna(r.get("hba1c_pct")) else "n/a"
        bmi_txt = f"{r['bmi']:.1f} kg/m²" if pd.notna(r.get("bmi")) else "n/a"
        st.markdown(
            f"""
            <div class="patient-card">
                <div class="label">Participant</div><div class="value"><b>{pid}</b></div>
                <div class="label">Gender / Treatment</div><div class="value">{gender_txt} · {treatment_txt}</div>
                <div class="label">Clinical context</div><div class="value">Age {age_txt} · HbA1c {hba1c_txt} · BMI {bmi_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Compact patient card above replaces raw metadata lines.
        # st.write(f"**Gender:** {r.get('gender', 'n/a')}")
        # st.write(f"**Treatment:** {r.get('treatment', 'n/a')}")
        if False and pd.notna(r.get("age_years")):
            st.write(f"**Age:** {r['age_years']:.0f} years")
        if False and pd.notna(r.get("hba1c_pct")):
            st.write(f"**HbA1c:** {r['hba1c_pct']:.1f} %")
        if False and pd.notna(r.get("bmi")):
            st.write(f"**BMI:** {r['bmi']:.1f} kg/m²")
    if not feature_row.empty:
        r = feature_row.iloc[0].to_dict()
        st.write(
            f"**Modality availability** — basal: {bool(r.get('basal_available', 1))}, "
            f"bolus: {bool(r.get('bolus_available', 1))}, "
            f"carb: {bool(r.get('carb_available', 1))}"
        )


# ---------------------------------------------------------------------------
# Build the forecast panel
# ---------------------------------------------------------------------------

# Recover the un-scaled glucose history from the z-scored X_dynamic
scalers_path = ROOT / C.SCALERS_JSON
with open(scalers_path) as fh:
    scalers_blob = json.load(fh)
glu_scaler = scalers_blob["dynamic"]["per_subject"]["glucose"][pid]
mu = glu_scaler["mean"]
sd = glu_scaler["std"]
glu_idx_in_dyn = feat_dyn.index("glucose")

x_dyn = test["X_dynamic"][sample_idx]                 # (24, 17)
x_stat = test["X_static"][sample_idx]                 # (n_stat + 1)
glu_history = x_dyn[:, glu_idx_in_dyn] * sd + mu      # (24,) mg/dL
last_glucose = float(glu_history[-1])

# y_true and y_pred at the three horizons
y_true_h = test["y"][sample_idx]                      # (3,)
# Model forward
with torch.no_grad():
    y_pred_h = model(
        torch.from_numpy(x_dyn[None]).float(),
        torch.from_numpy(x_stat[None]).float(),
    ).numpy()[0]

# Intervals from parquet (Mondrian-ACI alpha target = 0.10 == 90 % PI)
mask = (intervals["participant_id"] == pid) & (intervals["sample_idx"] == sample_idx)
intv = intervals[mask].set_index("horizon_min")
if len(intv) == 3:
    lo90 = [float(intv.loc[h, INTERVAL_LO_COL]) for h in HORIZONS]
    up90 = [float(intv.loc[h, INTERVAL_UP_COL]) for h in HORIZONS]
    widths90 = np.array(up90) - np.array(lo90)
else:
    lo90, up90 = [], []
    widths90 = np.array([np.nan, np.nan, np.nan])

primary_zone = risk_class(
    float(y_pred_h[0]),
    lo90[0] if len(lo90) == 3 else float(y_pred_h[0]),
    up90[0] if len(up90) == 3 else float(y_pred_h[0]),
)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown('<div class="metric-neutral">', unsafe_allow_html=True)
    st.metric("Current glucose", f"{last_glucose:.0f} mg/dL")
    st.markdown("</div>", unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown('<div class="metric-navy">', unsafe_allow_html=True)
    st.metric("+30 min forecast", f"{y_pred_h[0]:.0f} mg/dL")
    st.markdown("</div>", unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown('<div class="metric-risk">', unsafe_allow_html=True)
    st.metric("Risk status", risk_label(primary_zone))
    st.markdown("</div>", unsafe_allow_html=True)
with metric_cols[3]:
    st.markdown('<div class="metric-uncertainty">', unsafe_allow_html=True)
    st.metric("Mean ACI PI width", f"{np.nanmean(widths90):.0f} mg/dL")
    st.markdown("</div>", unsafe_allow_html=True)

# Plot
col_main, col_alert = st.columns([1, 0.001])

with col_main:
    st.markdown(
        """
        <div class="panel-card">
            <h3>Glucose Forecast with 90% Mondrian-ACI Interval</h3>
            <p class="panel-subtitle">Two-hour CGM history, multi-horizon forecast, held-out observation, and adaptive calibrated uncertainty band.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    # History: last 24 ticks (= 2 hours) ending at t=0
    t_hist = np.arange(-23 * 5, 5, 5)             # minutes relative to "now"
    ax.plot(t_hist, glu_history, color="#0F2E57", linewidth=1.65, label="Observed glucose")
    ax.scatter([0], [last_glucose], color="#0F2E57", zorder=5, s=42)

    # Forecast points
    t_fc = np.array([30, 60, 90])
    ax.plot(t_fc, y_pred_h, marker="o", linestyle="--",
            color="#D9A323", linewidth=1.8, label="Model forecast")
    ax.scatter(t_fc, y_true_h, marker="x", color="#d62728", s=70,
               linewidth=2, zorder=5, label="Observed (after-the-fact)")

    # 90 % PI band — connect (0, last) to (30, lower) to (60, lower) etc.
    if len(intv) == 3:
        lo90 = [intv.loc[h, INTERVAL_LO_COL] for h in HORIZONS]
        up90 = [intv.loc[h, INTERVAL_UP_COL] for h in HORIZONS]
        # Anchor band at t=0 to last_glucose for visual continuity
        ax.fill_between(
            [0, 30, 60, 90],
            [last_glucose] + lo90,
            [last_glucose] + up90,
            color="#123A6F", alpha=0.13, label=INTERVAL_LABEL,
        )

    ax.axhline(70, color="#C24141", linestyle=":", linewidth=0.9, alpha=0.68)
    ax.axhline(180, color="#D9A323", linestyle=":", linewidth=0.9, alpha=0.76)
    ax.text(-115, 64, "70 mg/dL", color="#9F2D2D", fontsize=8)
    ax.text(-115, 184, "180 mg/dL", color="#8A6515", fontsize=8)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Time relative to now (minutes; negative = past, positive = forecast)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlim(-120, 95)
    ax.set_ylim(min(40, glu_history.min() - 10),
                max(300, np.nanmax([up90[-1] if len(intv) == 3 else 200, glu_history.max() + 20])))
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    ax.grid(alpha=0.22, color="#B7C8BD")
    ax.tick_params(colors="#102A43")
    ax.xaxis.label.set_color("#102A43")
    ax.yaxis.label.set_color("#102A43")
    for spine in ax.spines.values():
        spine.set_color("#DDE7DF")
    st.markdown(fig_to_inline_html(fig, "Glucose forecast with uncertainty interval"), unsafe_allow_html=True)
    plt.close(fig)

    # Table of horizon-level numbers
    if len(intv) == 3:
        st.markdown(
            '<div class="section-title">Forecast Horizon Timeline</div><div class="small-muted">Each horizon shows the point forecast, adaptive calibrated 90% interval, and threshold flag.</div>',
            unsafe_allow_html=True,
        )
        risk_cols = st.columns(3)
        rows = []
        for h in HORIZONS:
            lo = float(intv.loc[h, INTERVAL_LO_COL])
            up = float(intv.loc[h, INTERVAL_UP_COL])
            yp = float(y_pred_h[HORIZONS.index(h)])
            zone = risk_class(yp, lo, up)
            cross_badge = (
                '<div class="threshold-chip">PI crosses threshold</div>'
                if (lo < 70 or up > 180)
                else '<div class="threshold-chip placeholder">PI crosses threshold</div>'
            )
            with risk_cols[HORIZONS.index(h)]:
                st.markdown(
                    f"""
                    <div class="risk-card {zone}">
                        <div class="risk-top">
                            <span class="horizon">+{h} min</span>
                            <span class="risk-badge">{risk_label(zone)}</span>
                        </div>
                        <div class="risk-values">
                            Forecast <b>{yp:.0f} mg/dL</b><br>
                            {INTERVAL_SHORT_LABEL} [{lo:.0f}, {up:.0f}]
                        </div>
                        {cross_badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            yt = float(y_true_h[HORIZONS.index(h)])
            rows.append({
                "Horizon": HORIZON_LABEL[h],
                "Forecast (mg/dL)": f"{yp:.1f}",
                "Forecast": f"{yp:.1f}",
                "90 % PI lower": f"{lo:.1f}",
                "90 % PI upper": f"{up:.1f}",
                "90% PI": f"[{lo:.1f}, {up:.1f}]",
                "PI width": f"{up - lo:.1f}",
                "Observed (held-out)": f"{yt:.1f}",
                "Observed": f"{yt:.1f}",
                "Model abs error": f"{abs(yt - yp):.1f}",
                "Model AE": f"{abs(yt - yp):.1f}",
            })
        st.markdown(render_forecast_table(rows), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="panel-card">
                <h3>Clarke Error Grid Safety Review</h3>
                <p class="panel-subtitle">Playback-only clinical interpretation: each forecast is compared with the held-out future glucose value.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        clarke_rows = []
        clarke_cols = st.columns(3)
        for h in HORIZONS:
            idx = HORIZONS.index(h)
            yp = float(y_pred_h[idx])
            yt = float(y_true_h[idx])
            z = clarke_zone_label(yt, yp)
            severity = clarke_zone_severity(z)
            desc = clarke_zone_description(z)
            with clarke_cols[idx]:
                st.markdown(
                    f"""
                    <div class="clarke-card zone-{z.lower()}">
                        <div class="horizon">+{h} min</div>
                        <div class="zone">Zone {z}</div>
                        <div><b>{severity}</b></div>
                        <div class="desc">Observed {yt:.0f} mg/dL -> Forecast {yp:.0f} mg/dL -> AE {abs(yt - yp):.1f} mg/dL</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            clarke_rows.append({
                "Horizon": HORIZON_LABEL[h],
                "Observed": f"{yt:.1f}",
                "Forecast": f"{yp:.1f}",
                "Clarke zone": f"Zone {z}",
                "Interpretation": desc,
                "Abs. error": f"{abs(yt - yp):.1f}",
            })
        st.markdown(render_clarke_table(clarke_rows), unsafe_allow_html=True)
        st.caption(
            "Clarke Error Grid Analysis requires the true future reference glucose. "
            "Therefore this panel is shown for HUPA playback/validation cases, not as a real-time treatment instruction."
        )

if False:
    st.markdown(
        '<div class="section-title">Risk Indicators</div><div class="small-muted">Threshold flags from forecast and 90% PI.</div>',
        unsafe_allow_html=True,
    )
    if len(intv) == 3:
        for h in HORIZONS:
            lo = float(intv.loc[h, INTERVAL_LO_COL])
            up = float(intv.loc[h, INTERVAL_UP_COL])
            yp = float(y_pred_h[HORIZONS.index(h)])
            yt = float(y_true_h[HORIZONS.index(h)])
            zone_pred = zone_of_value(yp)
            lower_hypo = lo < 70
            upper_hyper = up > 180
            if zone_pred == "hypo" or lower_hypo:
                st.error(
                    f"**+{h} min — Hypoglycaemia risk**\n\n"
                    f"Forecast {yp:.0f} mg/dL; 90 % PI [{lo:.0f}, {up:.0f}].",
                    icon="⚠️",
                )
            elif zone_pred == "hyper" or upper_hyper:
                st.warning(
                    f"**+{h} min — Hyperglycaemia risk**\n\n"
                    f"Forecast {yp:.0f} mg/dL; 90 % PI [{lo:.0f}, {up:.0f}].",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"**+{h} min — Time-in-range**\n\n"
                    f"Forecast {yp:.0f} mg/dL; 90 % PI [{lo:.0f}, {up:.0f}].",
                    icon="✅",
                )

st.markdown("---")

# ---------------------------------------------------------------------------
# Explanation panel — IG
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="panel-card">
        <h3>Integrated Gradients Explanation</h3>
        <p class="panel-subtitle">Attribution heatmap and feature rankings for the selected forecast horizon.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_h, col_btn = st.columns([1, 1])
with col_h:
    h_pick = st.radio(
        "Horizon to explain",
        HORIZONS,
        index=0,
        horizontal=True,
        format_func=lambda v: HORIZON_LABEL[v],
    )
with col_btn:
    run_now = st.button("Compute Integrated Gradients for this window",
                        help="Runs IG on demand (~5 seconds on CPU)")

if run_now:
    with st.spinner("Computing Integrated Gradients (50 Riemann steps) …"):
        h_idx = HORIZONS.index(h_pick)
        a_ig = compute_ig_for_window(model, x_dyn, x_stat, h_idx)   # (24, 17)

    # Heatmap
    hm = pd.DataFrame(np.abs(a_ig), columns=feat_dyn)
    importance = hm.sum(axis=0).sort_values(ascending=False)
    top_features = importance.head(10).index.tolist()
    hm_top = hm[top_features]                                       # (24, 10)

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    fig2.patch.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")
    im = ax2.imshow(hm_top.T.values, aspect="auto", cmap="YlGnBu")
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=9)
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xticklabels([f"t-{(23 - x) * 5}'" for x in range(0, 24, 4)], fontsize=9)
    ax2.set_xlabel("Lookback step")
    ax2.set_title(f"|IG| heatmap - top 10 features at horizon = {h_pick} min", color="#0F2E57")
    for spine in ax2.spines.values():
        spine.set_color("#DDE7DF")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    st.markdown(fig_to_inline_html(fig2, "Integrated Gradients feature-time heatmap"), unsafe_allow_html=True)
    plt.close(fig2)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            '<div class="panel-card"><h3>Top Contributing Features</h3><p class="panel-subtitle">Local attribution for this selected window.</p></div>',
            unsafe_allow_html=True,
        )
        rank_df = pd.DataFrame({
            "feature": importance.head(10).index,
            "share_pct": 100 * importance.head(10).values / importance.sum(),
        })
        rank_df["share_pct"] = rank_df["share_pct"].round(1)
        if len(intv) == 3:
            lo_pick = float(intv.loc[h_pick, INTERVAL_LO_COL])
            up_pick = float(intv.loc[h_pick, INTERVAL_UP_COL])
        else:
            lo_pick = None
            up_pick = None
        rule_text = explanation_text(h_pick, float(y_pred_h[h_idx]), lo_pick, up_pick, rank_df)
        st.markdown(f"<div class='xai-note'>{rule_text}</div>", unsafe_allow_html=True)

        if use_llm:
            selected_patient = clinical_df[clinical_df["participant_id"] == pid]
            patient_row = selected_patient.iloc[0] if not selected_patient.empty else None
            llm_prompt = build_llm_prompt(
                pid=pid,
                horizon=h_pick,
                last_glucose=last_glucose,
                y_pred=float(y_pred_h[h_idx]),
                lo=lo_pick,
                up=up_pick,
                rank_df=rank_df,
                patient_row=patient_row,
            )
            with st.spinner("Generating plain-language explanation ..."):
                if explanation_mode == "Ollama local":
                    llm_text, llm_error = call_ollama_explanation(llm_prompt, ollama_model)
                else:
                    llm_text, llm_error = call_llm_explanation(llm_prompt)
            if llm_text:
                st.markdown("**Plain-language explanation**")
                st.markdown(f"<div class='xai-note'>{llm_text}</div>", unsafe_allow_html=True)
            else:
                st.warning(f"LLM explanation unavailable. Showing deterministic explanation only. {llm_error}")
        fig_local, ax_local = plt.subplots(figsize=(5.8, 3.2))
        local_plot = rank_df.sort_values("share_pct", ascending=True)
        ax_local.barh(local_plot["feature"], local_plot["share_pct"], color="#123A6F", alpha=0.9)
        ax_local.set_xlabel("Attribution share (%)")
        ax_local.grid(axis="x", alpha=0.18, color="#B7C8BD")
        ax_local.set_facecolor("#FFFFFF")
        fig_local.patch.set_facecolor("#FFFFFF")
        for spine in ax_local.spines.values():
            spine.set_color("#DDE7DF")
        st.markdown(fig_to_inline_html(fig_local, "Local feature attribution ranking"), unsafe_allow_html=True)
        plt.close(fig_local)
        st.dataframe(rank_df, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown(
            '<div class="panel-card"><h3>Global Ranking</h3><p class="panel-subtitle">Average IG importance for the selected horizon.</p></div>',
            unsafe_allow_html=True,
        )
        g_imp = state["global_imp"]
        g_sub = g_imp[g_imp["horizon_min"] == h_pick].head(10)[
            ["feature", "importance_pct"]
        ].copy()
        g_sub["importance_pct"] = g_sub["importance_pct"].round(1)
        fig_global, ax_global = plt.subplots(figsize=(5.8, 3.2))
        global_plot = g_sub.sort_values("importance_pct", ascending=True)
        ax_global.barh(global_plot["feature"], global_plot["importance_pct"], color="#D9A323", alpha=0.92)
        ax_global.set_xlabel("Global importance (%)")
        ax_global.grid(axis="x", alpha=0.18, color="#B7C8BD")
        ax_global.set_facecolor("#FFFFFF")
        fig_global.patch.set_facecolor("#FFFFFF")
        for spine in ax_global.spines.values():
            spine.set_color("#DDE7DF")
        st.markdown(fig_to_inline_html(fig_global, "Global feature importance ranking"), unsafe_allow_html=True)
        plt.close(fig_global)
        st.dataframe(g_sub, hide_index=True, use_container_width=True)

    st.caption(
        "Brighter cells = larger absolute attribution. The right-most column "
        "(t-0) is the most recent observation; rows are sorted by total "
        "absolute attribution for this window. Compare the per-window "
        "ranking on the left with the global ranking on the right to see "
        "whether this prediction is driven by typical signals (matching "
        "global) or by an unusual signal pattern (diverging from global)."
    )
else:
    st.info(
        "Click **Compute Integrated Gradients for this window** to attribute "
        "the forecast to specific input features and lookback steps. "
        "Computation takes ~5 seconds on a single CPU core."
    )

st.markdown("---")
st.markdown(
    """
    <div>
        <span class="footer-chip"><b>Model</b> PersResid CNN-GRU-Attention</span>
        <span class="footer-chip"><b>Dataset</b> HUPA-UCM test split</span>
        <span class="footer-chip"><b>Uncertainty</b> Mondrian-ACI PI</span>
        <span class="footer-chip"><b>Clinical safety</b> Clarke EGA</span>
        <span class="footer-chip"><b>XAI</b> Integrated Gradients</span>
        <span class="footer-chip"><b>Unit</b> mg/dL</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if False:
    st.caption(
    "Implementation: `app/streamlit_app.py`  •  "
    "Model: `outputs/models/step6_hybrid_v2_pers_resid.pt`  •  "
    "ACI intervals: `outputs/tables/uq_aci_alpha_trajectory.parquet`  •  "
    "Static metadata: `data/processed/hupa_static_features.csv`.  "
    "Numbers throughout are in mg/dL on the held-out HUPA-UCM test split."
)

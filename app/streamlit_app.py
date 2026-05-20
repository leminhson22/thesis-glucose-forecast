"""Streamlit MVP for the proposed glucose-forecasting model.

A research demo, not a clinical tool. Loads the proposed
CNN-GRU-Attention + Persistence-Residual checkpoint (§7.6 of the thesis
report) and visualises, for any held-out test window:

  * the most recent 2 hours of CGM glucose;
  * the model's 30 / 60 / 90-minute forecast;
  * a calibrated 90 % prediction interval from §9 Mondrian Conformal CP;
  * a hypoglycaemia-risk indicator;
  * an Integrated Gradients explanation heat-map (§10) showing which
    inputs at which lookback steps drove the forecast.

The app loads everything from pre-computed artefacts under outputs/; the
model itself is loaded once and reused for on-demand IG when the selected
window is not in the case-study parquet.

Run locally with:

    streamlit run app/streamlit_app.py

This file deliberately is *not* a clinical-decision-support system. The
header banner in the UI states this and a research-only disclaimer is
shown alongside every forecast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from datasets import load_npz_splits  # noqa: E402
from eval_step6_v2 import load_variant_model  # noqa: E402
from explain import integrated_gradients_dyn, temporal_feature_heatmap  # noqa: E402
from run_step6_v2 import attach_pid_index_to_static, load_pid_scaler_table  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
FIG_DIR = ROOT / "outputs" / "figures"

HORIZONS = list(C.HORIZON_MINUTES)
HORIZON_LABEL = {30: "30 min", 60: "60 min", 90: "90 min"}
ZONE_COLOUR = {"hypo": "#d62728", "tir": "#2ca02c", "hyper": "#ff7f0e"}

st.set_page_config(
    page_title="Glucose-Forecast Research Demo (HUPA-UCM)",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

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

    intervals = pd.read_parquet(TABLES / "uq_conformal_intervals.parquet")
    cases = pd.read_parquet(TABLES / "xai_ig_case_studies.parquet")
    static_csv = pd.read_csv(ROOT / "data" / "processed" / "hupa_static_features.csv")
    global_imp = pd.read_csv(TABLES / "xai_ig_global_importance.csv")

    return {
        "splits": splits,
        "work": work,
        "model": model,
        "intervals": intervals,
        "cases": cases,
        "static_df": static_csv,
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


def fmt_zone(zone: str) -> str:
    return {
        "hypo": "Hypoglycaemia (<70 mg/dL)",
        "tir": "Time-in-range (70–180 mg/dL)",
        "hyper": "Hyperglycaemia (>180 mg/dL)",
    }[zone]


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
feat_dyn = state["feat_dyn"]
test = work["test"]
n_samples = test["X_dynamic"].shape[0]

st.title("Short-term glucose-forecast research demo")
st.caption(
    "Proposed model — CNN-GRU-Attention with Persistence-Residual Learning (§7.6) — "
    "wrapped with Mondrian Conformal Prediction intervals (§9) and Integrated-Gradients "
    "feature attributions (§10). Held-out HUPA-UCM test split."
)
st.warning(
    "**Research demo only.** Forecasts are not medical advice and must not replace "
    "clinician judgement.",
    icon="⚠️",
)

# Sidebar — sample picker
with st.sidebar:
    st.header("Pick a test window")

    all_pids = sorted(np.unique(test["pids"]).tolist())
    pid = st.selectbox("Patient", all_pids, index=min(13, len(all_pids) - 1))

    pid_mask = test["pids"] == pid
    pid_sample_idx = np.where(pid_mask)[0]
    if len(pid_sample_idx) == 0:
        st.error(f"No test windows for {pid}")
        st.stop()

    # Allow scrolling through this patient's test windows
    pos = st.slider(
        "Window index within this patient",
        min_value=0,
        max_value=len(pid_sample_idx) - 1,
        value=min(120, len(pid_sample_idx) - 1),
        step=1,
        help="Each step is one 5-minute forecast tick within this patient's test split.",
    )
    sample_idx = int(pid_sample_idx[pos])

    st.markdown("---")
    st.subheader("Patient summary")
    pat_row = static_df[static_df["participant_id"] == pid]
    if not pat_row.empty:
        r = pat_row.iloc[0].to_dict()
        st.write(f"**Treatment:** {r.get('treatment', 'n/a')}")
        if "age_years" in r:
            st.write(f"**Age:** {r['age_years']:.0f} years")
        if "hba1c_pct" in r:
            st.write(f"**HbA1c:** {r['hba1c_pct']:.1f} %")
        if "weight_kg" in r and "height_cm" in r:
            bmi = r["weight_kg"] / (r["height_cm"] / 100) ** 2
            st.write(f"**BMI:** {bmi:.1f} kg/m²")
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

# Intervals from parquet (Mondrian α=0.10 == 90 % PI)
mask = (intervals["participant_id"] == pid) & (intervals["sample_idx"] == sample_idx)
intv = intervals[mask].set_index("horizon_min")

# Plot
col_main, col_alert = st.columns([3, 1])

with col_main:
    st.subheader("Glucose history + forecast with 90 % Mondrian prediction interval")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # History: last 24 ticks (= 2 hours) ending at t=0
    t_hist = np.arange(-23 * 5, 5, 5)             # minutes relative to "now"
    ax.plot(t_hist, glu_history, color="black", linewidth=1.4, label="Observed glucose")
    ax.scatter([0], [last_glucose], color="black", zorder=5, s=40)

    # Forecast points
    t_fc = np.array([30, 60, 90])
    ax.plot(t_fc, y_pred_h, marker="o", linestyle="--",
            color="#1f77b4", linewidth=1.4, label="Model forecast")
    ax.scatter(t_fc, y_true_h, marker="x", color="#d62728", s=70,
               linewidth=2, zorder=5, label="Observed (after-the-fact)")

    # 90 % PI band — connect (0, last) to (30, lower) to (60, lower) etc.
    if len(intv) == 3:
        lo90 = [intv.loc[h, "lower_mondrian_a10"] for h in HORIZONS]
        up90 = [intv.loc[h, "upper_mondrian_a10"] for h in HORIZONS]
        # Anchor band at t=0 to last_glucose for visual continuity
        ax.fill_between(
            [0, 30, 60, 90],
            [last_glucose] + lo90,
            [last_glucose] + up90,
            color="#1f77b4", alpha=0.18, label="90 % Mondrian PI",
        )

    ax.axhline(70, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(180, color="#ff7f0e", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(-115, 64, "70 mg/dL", color="#d62728", fontsize=8)
    ax.text(-115, 184, "180 mg/dL", color="#ff7f0e", fontsize=8)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Time relative to now (minutes; negative = past, positive = forecast)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlim(-120, 95)
    ax.set_ylim(min(40, glu_history.min() - 10),
                max(300, np.nanmax([up90[-1] if len(intv) == 3 else 200, glu_history.max() + 20])))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    # Table of horizon-level numbers
    if len(intv) == 3:
        rows = []
        for h in HORIZONS:
            lo = float(intv.loc[h, "lower_mondrian_a10"])
            up = float(intv.loc[h, "upper_mondrian_a10"])
            yp = float(y_pred_h[HORIZONS.index(h)])
            yt = float(y_true_h[HORIZONS.index(h)])
            rows.append({
                "Horizon": HORIZON_LABEL[h],
                "Forecast (mg/dL)": f"{yp:.1f}",
                "90 % PI lower": f"{lo:.1f}",
                "90 % PI upper": f"{up:.1f}",
                "PI width": f"{up - lo:.1f}",
                "Observed (held-out)": f"{yt:.1f}",
                "Error |y−ŷ|": f"{abs(yt - yp):.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with col_alert:
    st.subheader("Risk indicators")
    if len(intv) == 3:
        for h in HORIZONS:
            lo = float(intv.loc[h, "lower_mondrian_a10"])
            up = float(intv.loc[h, "upper_mondrian_a10"])
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

st.subheader("Why this forecast — Integrated Gradients explanation")

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
    im = ax2.imshow(hm_top.T.values, aspect="auto", cmap="viridis")
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=9)
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xticklabels([f"t-{(23 - x) * 5}'" for x in range(0, 24, 4)], fontsize=9)
    ax2.set_xlabel("Lookback step")
    ax2.set_title(
        f"|IG| heat-map for this window — top-10 features at horizon = {h_pick} min"
    )
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    st.pyplot(fig2)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Top contributing features (this window)**")
        rank_df = pd.DataFrame({
            "feature": importance.head(10).index,
            "share_pct": 100 * importance.head(10).values / importance.sum(),
        })
        rank_df["share_pct"] = rank_df["share_pct"].round(1)
        st.dataframe(rank_df, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**Global ranking (for comparison)**")
        g_imp = state["global_imp"]
        g_sub = g_imp[g_imp["horizon_min"] == h_pick].head(10)[
            ["feature", "importance_pct"]
        ].copy()
        g_sub["importance_pct"] = g_sub["importance_pct"].round(1)
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
st.caption(
    "Implementation: `app/streamlit_app.py`  •  "
    "Model: `outputs/models/step6_hybrid_v2_pers_resid.pt`  •  "
    "Conformal intervals: `outputs/tables/uq_conformal_intervals.parquet`  •  "
    "Static metadata: `data/processed/hupa_static_features.csv`.  "
    "Numbers throughout are in mg/dL on the held-out HUPA-UCM test split."
)

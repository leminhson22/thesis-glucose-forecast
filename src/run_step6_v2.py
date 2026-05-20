"""Step 6 v2 — architecture ablation runner per report.md §8.7.

Each variant is a targeted change against the §8.6 baseline. The full
five-variant menu plus the baseline replay:

    baseline       — same config as the §8.6 step6_hybrid run (sanity check).
    no_moddrop     — disable training-time modality dropout.
    smaller        — halve hidden_dim/attn_dim/num_layers; combat overfit.
    no_attn        — drop cross-attention; concat last_h + static_emb.
    modal          — per-modality CNN branches (true multimodal fusion).
    pers_resid     — model outputs delta; final = last_glucose_mgdl + delta.

Usage::

    python src/run_step6_v2.py --variant no_moddrop
    python src/run_step6_v2.py --variant smaller
    python src/run_step6_v2.py --variant no_attn
    python src/run_step6_v2.py --variant modal
    python src/run_step6_v2.py --variant pers_resid
    python src/run_step6_v2.py --variant all          # sequential run of all five

Output per variant:

    outputs/logs/step6_hybrid_v2_<variant>.csv
    outputs/models/step6_hybrid_v2_<variant>.pt
    outputs/tables/step6_v2_<variant>_{per_horizon,per_zone,clarke,...}.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as C  # noqa: E402
from datasets import build_dataloaders, build_modality_groups, load_npz_splits  # noqa: E402
from evaluate import compact_summary  # noqa: E402
from losses import ZoneWeightedMSE  # noqa: E402
from models import (  # noqa: E402
    HYBRID_DEFAULTS,
    HybridCNNGRU,
    HybridCNNGRUNoAttn,
    HybridCNNGRUPersResid,
    HybridModalCNNGRU,
    count_parameters,
)
from train import TrainConfig, get_device, train_model  # noqa: E402


PROJECT_ROOT = _HERE.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Same winning loss config as the §8.6 baseline.
LOSS_KWARGS = dict(
    w_hypo=2.0, w_tir=1.0, w_hyper=1.5,
    horizon_weights=(1.5, 1.0, 1.0),
    hypo_under_detect_penalty=2.0,
)
STEP6_EARLY_STOP_PATIENCE = 5
STEP6_MODALITY_DROPOUT_P = 0.30

VARIANTS = ("baseline", "no_moddrop", "smaller", "no_attn", "modal", "pers_resid")


def load_pid_scaler_table(splits: dict) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build a per-patient (mean, std) tensor for glucose un-scaling.

    Returns (mean, std, pid_lookup). The lookup maps each participant_id
    string to a row index in the two tensors.
    """
    scalers_path = PROJECT_ROOT / C.SCALERS_JSON
    with open(scalers_path) as fh:
        s = json.load(fh)
    per_subject_glu = s["dynamic"]["per_subject"]["glucose"]
    pids = sorted(per_subject_glu.keys())
    pid_lookup = {pid: i for i, pid in enumerate(pids)}
    mean = torch.tensor([per_subject_glu[pid]["mean"] for pid in pids], dtype=torch.float32)
    std = torch.tensor([per_subject_glu[pid]["std"] for pid in pids], dtype=torch.float32)
    return mean, std, pid_lookup


def attach_pid_index_to_static(splits: dict, pid_lookup: dict[str, int]) -> dict:
    """Append a pid-index column (as float) to each split's X_static.

    The HybridCNNGRUPersResid model expects the last column of x_static
    to be a pid index it can use to look up per-subject scaler stats.
    """
    out = {k: splits[k] for k in ("feat_dyn", "feat_stat")}
    for name in ("train", "val", "test"):
        sp = dict(splits[name])
        pids = sp["pids"]
        idx_col = np.array([pid_lookup[p] for p in pids], dtype=np.float32).reshape(-1, 1)
        sp["X_static"] = np.concatenate([sp["X_static"], idx_col], axis=1)
        out[name] = sp
    return out


def make_model(
    variant: str,
    n_dynamic: int,
    n_static: int,
    feat_dyn: list[str],
    splits: dict,
):
    """Construct the variant model. Returns (model, modality_dropout_p)."""
    if variant in ("baseline", "no_moddrop"):
        model = HybridCNNGRU(n_dynamic=n_dynamic, n_static=n_static)
        mod_p = 0.0 if variant == "no_moddrop" else STEP6_MODALITY_DROPOUT_P
        return model, mod_p

    if variant == "smaller":
        model = HybridCNNGRU(
            n_dynamic=n_dynamic, n_static=n_static,
            hidden_dim=32, num_layers=1, attn_dim=24, attn_heads=2,
            static_embed_dim=16, head_hidden_dim=32, dropout=0.3,
        )
        return model, STEP6_MODALITY_DROPOUT_P

    if variant == "no_attn":
        model = HybridCNNGRUNoAttn(n_dynamic=n_dynamic, n_static=n_static)
        return model, STEP6_MODALITY_DROPOUT_P

    if variant == "modal":
        groups_named = build_modality_groups(feat_dyn)
        # Add a glucose-only branch (always present) and a "time/flags" group
        name_to_idx = {n: i for i, n in enumerate(feat_dyn)}
        groups = dict(groups_named)
        groups["glucose"] = []
        for n in ("glucose", "glucose_30m_mean", "glucose_60m_mean",
                 "glucose_120m_mean", "glucose_60m_std", "glucose_velocity"):
            if n in name_to_idx:
                groups["glucose"].append(name_to_idx[n])
        groups["time"] = []
        for n in ("hour_sin", "hour_cos", "glucose_low_cap", "basal_coverage_24h"):
            if n in name_to_idx:
                groups["time"].append(name_to_idx[n])
        # Drop empty groups
        groups = {k: v for k, v in groups.items() if v}
        model = HybridModalCNNGRU(
            n_dynamic=n_dynamic, n_static=n_static,
            modality_index_groups=groups,
        )
        return model, STEP6_MODALITY_DROPOUT_P

    if variant == "pers_resid":
        mean, std, pid_lookup = load_pid_scaler_table(splits)
        glu_idx = feat_dyn.index("glucose")
        # n_static here is the dataset width AFTER attaching the pid column,
        # but the base model expects (n_static - 1) because the wrapper slices
        # off the last column at forward time. Pass n_static - 1.
        model = HybridCNNGRUPersResid(
            n_dynamic=n_dynamic,
            n_static=n_static - 1,
            pid_glucose_mean=mean,
            pid_glucose_std=std,
            glucose_dyn_idx=glu_idx,
        )
        return model, STEP6_MODALITY_DROPOUT_P

    raise ValueError(f"unknown variant: {variant!r}")


def save_bundles(bundles: list[dict], variant: str) -> pd.DataFrame:
    prefix = f"step6_v2_{variant}"
    name_map = {
        "per_horizon": f"{prefix}_per_horizon.csv",
        "per_zone": f"{prefix}_per_zone.csv",
        "per_patient": f"{prefix}_per_patient.csv",
        "patient_averaged": f"{prefix}_patient_averaged.csv",
        "clarke_eg": f"{prefix}_clarke.csv",
    }
    by_key: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for b in bundles:
        for key, df in b.items():
            by_key[key].append(df)
    for key, frames in by_key.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(TABLES_DIR / name_map[key], index=False)
        print(f"[save] {name_map[key]}  rows={len(out)}")
    compact = pd.concat([compact_summary(b) for b in bundles], ignore_index=True)
    compact.to_csv(TABLES_DIR / f"{prefix}_summary.csv", index=False)
    print(f"[save] {prefix}_summary.csv  rows={len(compact)}")
    return compact


def run_one(variant: str, splits: dict, epochs: int, batch_size: int) -> pd.DataFrame:
    feat_dyn = splits["feat_dyn"]
    feat_stat_real = splits["feat_stat"]

    is_persres = (variant == "pers_resid")
    work_splits = attach_pid_index_to_static(
        splits, load_pid_scaler_table(splits)[2]
    ) if is_persres else splits

    n_dynamic = len(feat_dyn)
    n_static = work_splits["train"]["X_static"].shape[1]

    model, mod_p = make_model(
        variant=variant, n_dynamic=n_dynamic, n_static=n_static,
        feat_dyn=feat_dyn, splits=work_splits,
    )
    n_params = count_parameters(model)

    loaders = build_dataloaders(
        work_splits, batch_size=batch_size, num_workers=0,
        train_modality_dropout_p=float(mod_p),
        seed=C.SEED,
    )
    loss_fn = ZoneWeightedMSE(**LOSS_KWARGS)
    cfg = TrainConfig(
        epochs=epochs,
        early_stopping_patience=min(STEP6_EARLY_STOP_PATIENCE, max(2, epochs // 3)),
        lr_scheduler_patience=min(3, max(1, epochs // 5)),
    )
    run_tag = f"step6_hybrid_v2_{variant}"

    print(
        f"\n[{run_tag}] {type(model).__name__}  params={n_params:,}  "
        f"mod_dropout_p={mod_p}  n_dyn={n_dynamic}  n_stat={n_static}"
    )

    t0 = time.time()
    result = train_model(
        model=model, loaders=loaders, loss_fn=loss_fn, cfg=cfg,
        run_tag=run_tag, logs_dir=LOGS_DIR, models_dir=MODELS_DIR, verbose=True,
    )
    print(
        f"[{run_tag}] done in {time.time()-t0:.1f}s. "
        f"Best epoch={result['best_epoch']}, "
        f"val pat-avg MAE={result['best_val_pat_avg_mae']:.3f}"
    )

    compact = save_bundles(
        [result["final"]["val"]["bundle"], result["final"]["test"]["bundle"]],
        variant=variant,
    )

    show = [
        "model", "split", "horizon_min", "mae", "rmse",
        "mae_pat_avg", "clarke_pct_A", "clarke_pct_D",
    ]
    show = [c for c in show if c in compact.columns]
    print(f"\n=== {run_tag} compact summary ===")
    print(compact[show].to_string(index=False))
    return compact


def main(variant: str, epochs: int) -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if variant not in VARIANTS and variant != "all":
        raise ValueError(f"variant must be one of {VARIANTS} or 'all'; got {variant!r}")

    npz_path = PROJECT_ROOT / C.SEQUENCES_NPZ
    print(f"[load] {npz_path}")
    splits = load_npz_splits(npz_path)
    sizes = {k: int(splits[k]["y"].shape[0]) for k in ("train", "val", "test")}
    print(f"[load] split counts: {sizes}")

    variants_to_run = list(VARIANTS) if variant == "all" else [variant]
    summaries: list[pd.DataFrame] = []
    t_global = time.time()
    for v in variants_to_run:
        summaries.append(run_one(v, splits, epochs=epochs, batch_size=128))
        print(f"\n[overall elapsed] {time.time()-t_global:.1f}s after variant '{v}'")

    if len(summaries) > 1:
        all_compact = pd.concat(summaries, ignore_index=True)
        all_compact.to_csv(TABLES_DIR / "step6_v2_all_compact.csv", index=False)
        print(f"\n[save] step6_v2_all_compact.csv  rows={len(all_compact)}")

    print(f"\n[done] total elapsed = {time.time()-t_global:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="no_moddrop",
                    help=f"one of {VARIANTS} or 'all'")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    raise SystemExit(main(variant=args.variant, epochs=args.epochs))

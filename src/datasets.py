"""PyTorch Dataset / DataLoader builder for HUPA-UCM glucose forecasting (Phase C).

Loads sequences from ``data/processed/hupa_5min_sequences.npz`` and exposes
per-split iterators. The dataset returns ``(X_dynamic, X_static, y)`` float32
tensors; participant IDs are stored as ``.pids`` aligned row-by-row, so
evaluation utilities can join them after gathering predictions in dataset
order (use ``shuffle=False`` on val/test for this to hold).

The ``modality_dropout_p`` hook is wired in C.1 but defaulted to ``0.0``
(off). Phase C.3 will enable it for training the proposed hybrid model;
the same modality-group definition is reused by the M0--M4 deployment-tier
evaluator in C.4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config as C
except ImportError:
    import config as C  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Modality grouping for dropout (C.3) and tier evaluation (C.4)
# ---------------------------------------------------------------------------

def build_modality_groups(feature_names_dynamic: Sequence[str]) -> dict[str, list[int]]:
    """Map deployment-relevant modality names to feature indices in X_dynamic.

    Groups follow the deployment-tier strategy: only modalities that can
    plausibly be absent or unreliable in a real-world deployment are
    dropout candidates. ``glucose`` itself, time encodings, sensor flags,
    and basal coverage are NEVER dropped.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names_dynamic)}
    groups: dict[str, list[int]] = {
        "insulin": [],
        "carbs": [],
        "activity": [],
        "heart_rate": [],
    }
    for n in ("basal_rate", "bolus_60m_sum", "insulin_on_board"):
        if n in name_to_idx:
            groups["insulin"].append(name_to_idx[n])
    for n in ("carbs_on_board",):
        if n in name_to_idx:
            groups["carbs"].append(name_to_idx[n])
    for n in ("steps_150m_sum",):
        if n in name_to_idx:
            groups["activity"].append(name_to_idx[n])
    for n in ("heart_rate", "heart_rate_30m_mean"):
        if n in name_to_idx:
            groups["heart_rate"].append(name_to_idx[n])
    return groups


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HUPASequenceDataset(Dataset):
    """Wraps one split of the HUPA NPZ as a ``(X_dyn, X_stat, y)`` iterable.

    Parameters
    ----------
    modality_dropout_p
        Probability of zero-ing each modality group's features (independently)
        on every ``__getitem__`` call. ``0.0`` disables dropout. Used at C.3+.
    rng
        ``np.random.Generator`` used for modality dropout. Pass a seeded
        generator for reproducible runs.
    """

    def __init__(
        self,
        X_dynamic: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        pids: np.ndarray,
        feature_names_dynamic: Sequence[str] | None = None,
        feature_names_static: Sequence[str] | None = None,
        modality_dropout_p: float = 0.0,
        modality_groups: dict[str, list[int]] | None = None,
        rng: np.random.Generator | None = None,
    ):
        n = X_dynamic.shape[0]
        if X_static.shape[0] != n or y.shape[0] != n or pids.shape[0] != n:
            raise ValueError(
                f"row count mismatch: dyn={n}, stat={X_static.shape[0]}, "
                f"y={y.shape[0]}, pids={pids.shape[0]}"
            )
        self.X_dyn = X_dynamic.astype(np.float32, copy=False)
        self.X_stat = X_static.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.pids = np.asarray(pids).astype(str)
        self.feature_names_dynamic = list(feature_names_dynamic) if feature_names_dynamic else None
        self.feature_names_static = list(feature_names_static) if feature_names_static else None
        self.modality_dropout_p = float(modality_dropout_p)
        if self.modality_dropout_p > 0:
            if self.feature_names_dynamic is None:
                raise ValueError("feature_names_dynamic required when modality_dropout_p > 0")
            self.modality_groups = modality_groups or build_modality_groups(self.feature_names_dynamic)
        else:
            self.modality_groups = modality_groups or {}
        self._rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return self.X_dyn.shape[0]

    def __getitem__(self, idx: int):
        x_dyn = self.X_dyn[idx]
        x_stat = self.X_stat[idx]
        if self.modality_dropout_p > 0 and self.modality_groups:
            x_dyn = x_dyn.copy()
            for indices in self.modality_groups.values():
                if not indices:
                    continue
                if self._rng.random() < self.modality_dropout_p:
                    x_dyn[:, indices] = 0.0
        return (
            torch.from_numpy(x_dyn),
            torch.from_numpy(x_stat),
            torch.from_numpy(self.y[idx]),
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_npz_splits(npz_path: str | Path) -> dict:
    """Load the NPZ once and slice into train/val/test dicts ready for ``Dataset``."""
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    split = d["split"].astype(str)
    feat_dyn = [str(s) for s in d["feature_names_dynamic"]]
    feat_stat = [str(s) for s in d["feature_names_static"]]
    out = {"feat_dyn": feat_dyn, "feat_stat": feat_stat}
    for name in ("train", "val", "test"):
        mask = split == name
        out[name] = {
            "X_dynamic": d["X_dynamic"][mask].astype(np.float32),
            "X_static": d["X_static"][mask].astype(np.float32),
            "y": d["y"][mask].astype(np.float32),
            "pids": d["participant_ids"][mask].astype(str),
        }
    return out


def build_dataloaders(
    splits: dict,
    batch_size: int = 128,
    num_workers: int = 0,
    train_modality_dropout_p: float = 0.0,
    seed: int = C.SEED,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Build train (shuffle=True) and val/test (shuffle=False) ``DataLoader``s.

    The shuffle generator and (optional) modality-dropout rng are both
    seeded from ``seed`` so the training run is reproducible.
    """
    feat_dyn = splits["feat_dyn"]
    feat_stat = splits["feat_stat"]
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(int(seed))
    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        sp = splits[name]
        rng = np.random.default_rng(seed) if name == "train" else None
        ds = HUPASequenceDataset(
            sp["X_dynamic"], sp["X_static"], sp["y"], sp["pids"],
            feature_names_dynamic=feat_dyn,
            feature_names_static=feat_stat,
            modality_dropout_p=train_modality_dropout_p if name == "train" else 0.0,
            rng=rng,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            drop_last=False,
            generator=shuffle_gen if name == "train" else None,
        )
    return loaders

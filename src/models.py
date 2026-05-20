"""Recurrent baseline models for HUPA-UCM glucose forecasting (Phase C.1).

Both ``LSTM`` and ``GRU`` regressors share the same I/O contract so the
training loop is model-agnostic::

    forward(X_dynamic, X_static) -> y_pred

where:

* ``X_dynamic``: ``(B, lookback_steps=24, n_dyn_features=17)``
* ``X_static``:  ``(B, n_stat_features=16)``
* ``y_pred``:    ``(B, n_horizons=3)`` — mg/dL at 30 / 60 / 90 min

Architecture (identical structure for LSTM and GRU; only the recurrent
cell differs)::

    [X_dyn] -> RNN(hidden_dim, num_layers) -> last_hidden(hidden_dim)
                                                  |
    [X_stat] -> MLP --------------------> static_emb(static_embed_dim)
                                                  |
                          concat([last_hidden, static_emb])
                                                  |
                          head: Linear -> ReLU -> Linear(n_horizons)

The static branch follows the same input contract as the Phase A/B
flattened baselines (see ``src/baselines.py::flatten_window`` — Ridge,
Random Forest, and HistGB all consume both X_dynamic and X_static after
feature selection). This makes Phase C.1 an apples-to-apples comparison
with the tree baselines.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

try:
    from . import config as C
except ImportError:
    import config as C  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Defaults sized for the (24, 17) + (16,) -> (3,) HUPA problem with ~70k
# training samples. Sensible starting point; tunable from C.2 onward.
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    static_embed_dim=32,
    head_hidden_dim=64,
    bidirectional=False,  # glucose forecasting is causal; no peek at future
)


class _StaticBranch(nn.Module):
    """Small MLP that embeds the static-feature vector."""

    def __init__(self, n_static: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x_stat: torch.Tensor) -> torch.Tensor:
        return self.net(x_stat)


class _MultiHorizonHead(nn.Module):
    """Dense block mapping fused features to ``(B, n_horizons)`` mg/dL output."""

    def __init__(self, in_dim: int, hidden_dim: int, n_horizons: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class RecurrentRegressor(nn.Module):
    """Generic LSTM/GRU encoder + static MLP branch + multi-horizon head."""

    def __init__(
        self,
        rnn_type: Literal["lstm", "gru"],
        n_dynamic: int,
        n_static: int,
        n_horizons: int = len(C.HORIZON_MINUTES),
        hidden_dim: int = DEFAULTS["hidden_dim"],
        num_layers: int = DEFAULTS["num_layers"],
        dropout: float = DEFAULTS["dropout"],
        static_embed_dim: int = DEFAULTS["static_embed_dim"],
        head_hidden_dim: int = DEFAULTS["head_hidden_dim"],
        bidirectional: bool = DEFAULTS["bidirectional"],
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru'; got {rnn_type!r}")
        self.rnn_type = rnn_type
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=int(n_dynamic),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bool(bidirectional),
        )
        rnn_out_dim = int(hidden_dim) * (2 if bidirectional else 1)
        self.static_branch = _StaticBranch(
            n_static, embed_dim=static_embed_dim, dropout=dropout,
        )
        self.head = _MultiHorizonHead(
            in_dim=rnn_out_dim + static_embed_dim,
            hidden_dim=head_hidden_dim,
            n_horizons=n_horizons,
            dropout=dropout,
        )
        self.config = dict(
            rnn_type=rnn_type,
            n_dynamic=int(n_dynamic),
            n_static=int(n_static),
            n_horizons=int(n_horizons),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            static_embed_dim=int(static_embed_dim),
            head_hidden_dim=int(head_hidden_dim),
            bidirectional=bool(bidirectional),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_dyn)
        last = out[:, -1, :]                        # (B, rnn_out_dim)
        stat = self.static_branch(x_stat)           # (B, static_embed_dim)
        fused = torch.cat([last, stat], dim=-1)
        return self.head(fused)


def lstm_regressor(n_dynamic: int, n_static: int, **kwargs) -> RecurrentRegressor:
    """Convenience constructor: ``RecurrentRegressor('lstm', ...)``."""
    return RecurrentRegressor(rnn_type="lstm", n_dynamic=n_dynamic, n_static=n_static, **kwargs)


def gru_regressor(n_dynamic: int, n_static: int, **kwargs) -> RecurrentRegressor:
    """Convenience constructor: ``RecurrentRegressor('gru', ...)``."""
    return RecurrentRegressor(rnn_type="gru", n_dynamic=n_dynamic, n_static=n_static, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Step 6 hybrid: multi-kernel 1D-CNN + GRU + static cross-attention
# ---------------------------------------------------------------------------

HYBRID_DEFAULTS = dict(
    cnn_channels_per_kernel=16,
    cnn_kernels=(3, 5, 7),
    hidden_dim=64,
    num_layers=2,
    static_embed_dim=32,
    attn_dim=48,
    attn_heads=4,
    head_hidden_dim=64,
    dropout=0.3,
)


class _MultiKernelCNN1d(nn.Module):
    """Three parallel ``Conv1d`` branches with kernels 3 / 5 / 7 over time.

    Per `reports/model_choice_rationale.md` §3, the kernels correspond to
    15 / 25 / 35-minute receptive fields on the 5-minute HUPA grid, matching
    the rolling-mean spans the gradient-boosted tree baselines consume as
    engineered features. The concatenated activation is the CNN-front-end
    feature representation handed to the GRU reader.
    """

    def __init__(
        self,
        n_in: int,
        n_per_kernel: int,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(n_in, n_per_kernel, kernel_size=int(k), padding=int(k) // 2)
            for k in kernels
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(n_per_kernel) * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F_in) → channels-first for Conv1d
        x = x.transpose(1, 2)                         # (B, F_in, T)
        outs = [self.act(c(x)) for c in self.convs]
        y = torch.cat(outs, dim=1)                    # (B, n_per_kernel*K, T)
        y = self.dropout(y)
        return y.transpose(1, 2)                      # back to (B, T, out_dim)


class _StaticCrossAttention(nn.Module):
    """Single-query multi-head attention with the static patient embedding
    as the query and the GRU output sequence as keys/values.

    The static embedding provides per-patient conditioning: the attention
    weights over time steps are a function of the patient's demographics,
    HbA1c, and treatment type, so the network can learn that, e.g., a
    high-HbA1c patient needs more weight on the most recent two lags while
    a low-HbA1c patient can integrate longer trajectories.
    """

    def __init__(
        self,
        static_dim: int,
        seq_dim: int,
        attn_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.q_proj = nn.Linear(static_dim, attn_dim)
        self.k_proj = nn.Linear(seq_dim, attn_dim)
        self.v_proj = nn.Linear(seq_dim, attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.out_dim = int(attn_dim)

    def forward(self, static_emb: torch.Tensor, seq_out: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(static_emb).unsqueeze(1)      # (B, 1, attn_dim)
        k = self.k_proj(seq_out)                      # (B, T, attn_dim)
        v = self.v_proj(seq_out)                      # (B, T, attn_dim)
        attended, _ = self.attn(q, k, v, need_weights=False)
        return attended.squeeze(1)                    # (B, attn_dim)


class HybridCNNGRU(nn.Module):
    """Step 6 proposed model.

    Forward pipeline::

        X_dyn  (B, T, F_dyn) ──► MultiKernel-CNN  ──► (B, T, C_cnn)
                                                     │
                                                     ▼
                                               GRU(h, L)
                                                     │
                            ┌────────────────────────┤
                            │                        │
        X_stat (B, F_stat) ─► static MLP ──►  cross-attn (Q=stat, KV=seq)
                            │                        │
                            ▼                        ▼
                        concat(last_h, attended, stat_emb)
                                                     │
                                                     ▼
                                               head ──► (B, n_horizons)

    Param budget at the default config (n_dyn=17, n_stat=16): roughly 70k,
    comparable to the C.1/C.2 GRU (49k) but with the additional CNN
    front-end and cross-attention block. See
    `reports/model_choice_rationale.md` §4 for the architectural rationale.
    """

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        n_horizons: int = len(C.HORIZON_MINUTES),
        cnn_channels_per_kernel: int = HYBRID_DEFAULTS["cnn_channels_per_kernel"],
        cnn_kernels: tuple[int, ...] = HYBRID_DEFAULTS["cnn_kernels"],
        hidden_dim: int = HYBRID_DEFAULTS["hidden_dim"],
        num_layers: int = HYBRID_DEFAULTS["num_layers"],
        static_embed_dim: int = HYBRID_DEFAULTS["static_embed_dim"],
        attn_dim: int = HYBRID_DEFAULTS["attn_dim"],
        attn_heads: int = HYBRID_DEFAULTS["attn_heads"],
        head_hidden_dim: int = HYBRID_DEFAULTS["head_hidden_dim"],
        dropout: float = HYBRID_DEFAULTS["dropout"],
    ):
        super().__init__()
        self.cnn = _MultiKernelCNN1d(
            n_in=int(n_dynamic),
            n_per_kernel=int(cnn_channels_per_kernel),
            kernels=tuple(cnn_kernels),
            dropout=dropout,
        )
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.static_branch = _StaticBranch(
            int(n_static), embed_dim=int(static_embed_dim), dropout=dropout,
        )
        self.cross_attn = _StaticCrossAttention(
            static_dim=int(static_embed_dim),
            seq_dim=int(hidden_dim),
            attn_dim=int(attn_dim),
            n_heads=int(attn_heads),
            dropout=dropout,
        )
        fuse_dim = int(hidden_dim) + int(attn_dim) + int(static_embed_dim)
        self.head = _MultiHorizonHead(
            in_dim=fuse_dim,
            hidden_dim=int(head_hidden_dim),
            n_horizons=int(n_horizons),
            dropout=dropout,
        )
        self.config = dict(
            model_type="hybrid_cnn_gru",
            n_dynamic=int(n_dynamic),
            n_static=int(n_static),
            n_horizons=int(n_horizons),
            cnn_channels_per_kernel=int(cnn_channels_per_kernel),
            cnn_kernels=tuple(int(k) for k in cnn_kernels),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            static_embed_dim=int(static_embed_dim),
            attn_dim=int(attn_dim),
            attn_heads=int(attn_heads),
            head_hidden_dim=int(head_hidden_dim),
            dropout=float(dropout),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x_dyn)                     # (B, T, C_cnn)
        gru_out, _ = self.gru(cnn_out)                # (B, T, H)
        last_h = gru_out[:, -1, :]                    # (B, H)
        stat_emb = self.static_branch(x_stat)         # (B, S)
        attended = self.cross_attn(stat_emb, gru_out)  # (B, A)
        fused = torch.cat([last_h, attended, stat_emb], dim=-1)
        return self.head(fused)


def hybrid_cnn_gru(n_dynamic: int, n_static: int, **kwargs) -> HybridCNNGRU:
    """Convenience constructor matching ``lstm_regressor`` / ``gru_regressor`` style."""
    return HybridCNNGRU(n_dynamic=n_dynamic, n_static=n_static, **kwargs)


# ---------------------------------------------------------------------------
# Proposed thesis main model — CNN-LSTM dynamic branch + MLP static branch
# ---------------------------------------------------------------------------


class HybridStaticDynamic(nn.Module):
    """Proposed thesis main model — clean two-branch hybrid.

    Architecture per SKILL.md §5.2 "Static context branch + temporal encoder
    branch + fusion" pattern, kept deliberately simple:

    * **Dynamic branch:** two stacked ``Conv1d`` layers extract local
      temporal features from the 17-channel lookback window, followed by a
      two-layer ``LSTM`` that summarises the resulting sequence into a
      single hidden state. The LSTM cell is preferred over GRU here as a
      neutral default following the canonical CGM hybrid paper
      (Alkanhel et al. 2024 ``CNN-LSTM``).
    * **Static branch:** two-layer MLP embeds the 16 patient-level static
      features into a low-dimensional vector.
    * **Fusion:** late concatenation of the two branch outputs feeds a
      two-layer dense head that emits three horizon predictions in mg/dL.

    No persistence-residual, no modality dropout, no zone-weighted loss,
    no cross-attention. Trained with vanilla multi-horizon MSE.
    """

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        n_horizons: int = len(C.HORIZON_MINUTES),
        cnn_filters: tuple[int, int] = (64, 128),
        cnn_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        static_hidden: int = 64,
        static_embed: int = 32,
        head_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        if len(cnn_filters) != 2:
            raise ValueError("cnn_filters must have exactly two entries")
        c1, c2 = int(cnn_filters[0]), int(cnn_filters[1])
        pad = int(cnn_kernel) // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(int(n_dynamic), c1, kernel_size=int(cnn_kernel), padding=pad),
            nn.GELU(),
            nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=int(cnn_kernel), padding=pad),
            nn.GELU(),
            nn.BatchNorm1d(c2),
        )
        lstm_drop = float(dropout) if int(lstm_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=lstm_drop,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(int(n_static), int(static_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(static_hidden), int(static_embed)),
            nn.GELU(),
        )
        fuse_dim = int(lstm_hidden) + int(static_embed)
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), int(n_horizons)),
        )
        self.config = dict(
            model_type="hybrid_static_dynamic",
            n_dynamic=int(n_dynamic),
            n_static=int(n_static),
            n_horizons=int(n_horizons),
            cnn_filters=(c1, c2),
            cnn_kernel=int(cnn_kernel),
            lstm_hidden=int(lstm_hidden),
            lstm_layers=int(lstm_layers),
            static_hidden=int(static_hidden),
            static_embed=int(static_embed),
            head_hidden=int(head_hidden),
            dropout=float(dropout),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        # x_dyn: (B, T, F) -> Conv1d expects (B, F, T)
        x = x_dyn.transpose(1, 2)
        x = self.cnn(x)                      # (B, c2, T)
        x = x.transpose(1, 2)                # (B, T, c2)
        h_seq, _ = self.lstm(x)              # (B, T, lstm_hidden)
        dyn_h = h_seq[:, -1, :]              # last hidden state
        stat_emb = self.static_mlp(x_stat)   # (B, static_embed)
        fused = torch.cat([dyn_h, stat_emb], dim=-1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Step 6 ablation variants — see report.md §8.7 architecture audit
# ---------------------------------------------------------------------------


class HybridCNNGRUNoAttn(nn.Module):
    """Step 6 v2 ablation: drop cross-attention; concat last_h + static_emb only.

    Tests whether the patient-conditioned cross-attention block contributes
    to test performance once the multi-kernel CNN + GRU + static MLP path
    is already in place. The hypothesis is that the cross-attention's
    query (static patient embedding) is constant within one inference call
    and cannot react to recent rate-of-change, so it may hurt rather than
    help long-horizon CG-EGA EP.
    """

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        n_horizons: int = len(C.HORIZON_MINUTES),
        cnn_channels_per_kernel: int = HYBRID_DEFAULTS["cnn_channels_per_kernel"],
        cnn_kernels: tuple[int, ...] = HYBRID_DEFAULTS["cnn_kernels"],
        hidden_dim: int = HYBRID_DEFAULTS["hidden_dim"],
        num_layers: int = HYBRID_DEFAULTS["num_layers"],
        static_embed_dim: int = HYBRID_DEFAULTS["static_embed_dim"],
        head_hidden_dim: int = HYBRID_DEFAULTS["head_hidden_dim"],
        dropout: float = HYBRID_DEFAULTS["dropout"],
    ):
        super().__init__()
        self.cnn = _MultiKernelCNN1d(
            n_in=int(n_dynamic),
            n_per_kernel=int(cnn_channels_per_kernel),
            kernels=tuple(cnn_kernels),
            dropout=dropout,
        )
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.static_branch = _StaticBranch(
            int(n_static), embed_dim=int(static_embed_dim), dropout=dropout,
        )
        fuse_dim = int(hidden_dim) + int(static_embed_dim)
        self.head = _MultiHorizonHead(
            in_dim=fuse_dim,
            hidden_dim=int(head_hidden_dim),
            n_horizons=int(n_horizons),
            dropout=dropout,
        )
        self.config = dict(
            model_type="hybrid_cnn_gru_no_attn",
            n_dynamic=int(n_dynamic),
            n_static=int(n_static),
            n_horizons=int(n_horizons),
            cnn_channels_per_kernel=int(cnn_channels_per_kernel),
            cnn_kernels=tuple(int(k) for k in cnn_kernels),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            static_embed_dim=int(static_embed_dim),
            head_hidden_dim=int(head_hidden_dim),
            dropout=float(dropout),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x_dyn)
        gru_out, _ = self.gru(cnn_out)
        last_h = gru_out[:, -1, :]
        stat_emb = self.static_branch(x_stat)
        fused = torch.cat([last_h, stat_emb], dim=-1)
        return self.head(fused)


class _ModalCNN1d(nn.Module):
    """Per-modality multi-kernel CNN branches with late-concat fusion.

    Splits ``x_dyn`` by modality groups (glucose, insulin, activity,
    heart_rate, time/flags) and runs an independent multi-kernel CNN on
    each group. The resulting per-group sequences are concatenated along
    the channel dimension before the GRU reader — this is the
    architectural correction to Finding 1 of §8.7: separate per-modality
    branches replace the single shared Conv1d over all 17 features.
    """

    def __init__(
        self,
        modality_index_groups: dict[str, list[int]],
        n_per_kernel: int,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.modality_index_groups = {k: list(v) for k, v in modality_index_groups.items() if v}
        if not self.modality_index_groups:
            raise ValueError("modality_index_groups is empty")
        self.branches = nn.ModuleDict({
            name: _MultiKernelCNN1d(
                n_in=len(indices),
                n_per_kernel=n_per_kernel,
                kernels=kernels,
                dropout=dropout,
            )
            for name, indices in self.modality_index_groups.items()
        })
        self.out_dim = sum(b.out_dim for b in self.branches.values())

    def forward(self, x_dyn: torch.Tensor) -> torch.Tensor:
        outs = []
        for name, indices in self.modality_index_groups.items():
            sub = x_dyn[:, :, indices]
            outs.append(self.branches[name](sub))
        return torch.cat(outs, dim=-1)


class HybridModalCNNGRU(nn.Module):
    """Step 6 v2 ablation: true multimodal — per-modality CNN branches.

    Replaces ``_MultiKernelCNN1d`` (a single shared Conv1d over all 17
    features) with parallel per-modality CNNs and concatenates their
    outputs before the GRU reader. Static patient cross-attention is
    retained — this variant isolates Finding 1 ("multivariate, not
    multimodal") from the other architectural findings.
    """

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        modality_index_groups: dict[str, list[int]],
        n_horizons: int = len(C.HORIZON_MINUTES),
        cnn_channels_per_kernel: int = 8,  # halved per branch so total channels ≈ baseline
        cnn_kernels: tuple[int, ...] = HYBRID_DEFAULTS["cnn_kernels"],
        hidden_dim: int = HYBRID_DEFAULTS["hidden_dim"],
        num_layers: int = HYBRID_DEFAULTS["num_layers"],
        static_embed_dim: int = HYBRID_DEFAULTS["static_embed_dim"],
        attn_dim: int = HYBRID_DEFAULTS["attn_dim"],
        attn_heads: int = HYBRID_DEFAULTS["attn_heads"],
        head_hidden_dim: int = HYBRID_DEFAULTS["head_hidden_dim"],
        dropout: float = HYBRID_DEFAULTS["dropout"],
    ):
        super().__init__()
        self.cnn = _ModalCNN1d(
            modality_index_groups=modality_index_groups,
            n_per_kernel=int(cnn_channels_per_kernel),
            kernels=tuple(cnn_kernels),
            dropout=dropout,
        )
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.static_branch = _StaticBranch(
            int(n_static), embed_dim=int(static_embed_dim), dropout=dropout,
        )
        self.cross_attn = _StaticCrossAttention(
            static_dim=int(static_embed_dim),
            seq_dim=int(hidden_dim),
            attn_dim=int(attn_dim),
            n_heads=int(attn_heads),
            dropout=dropout,
        )
        fuse_dim = int(hidden_dim) + int(attn_dim) + int(static_embed_dim)
        self.head = _MultiHorizonHead(
            in_dim=fuse_dim,
            hidden_dim=int(head_hidden_dim),
            n_horizons=int(n_horizons),
            dropout=dropout,
        )
        self.config = dict(
            model_type="hybrid_modal_cnn_gru",
            n_dynamic=int(n_dynamic),
            n_static=int(n_static),
            modality_index_groups={k: list(v) for k, v in modality_index_groups.items()},
            n_horizons=int(n_horizons),
            cnn_channels_per_kernel=int(cnn_channels_per_kernel),
            cnn_kernels=tuple(int(k) for k in cnn_kernels),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            static_embed_dim=int(static_embed_dim),
            attn_dim=int(attn_dim),
            attn_heads=int(attn_heads),
            head_hidden_dim=int(head_hidden_dim),
            dropout=float(dropout),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x_dyn)
        gru_out, _ = self.gru(cnn_out)
        last_h = gru_out[:, -1, :]
        stat_emb = self.static_branch(x_stat)
        attended = self.cross_attn(stat_emb, gru_out)
        fused = torch.cat([last_h, attended, stat_emb], dim=-1)
        return self.head(fused)


class HybridCNNGRUSeq2Seq(nn.Module):
    """Step 6 v3 variant: seq2seq trajectory predictor.

    Outputs a full ``(B, n_traj_steps)`` trajectory covering t+5 through
    t+90 in 5-minute increments (default ``n_traj_steps=18``). The
    training signal is the trajectory itself plus a rate-of-change term
    (see :class:`losses.TrajectoryLoss`); the supervision on every
    intermediate step is what gives the model a direct gradient on
    predicted direction and rate, which is the structural fix for the
    §8.6 / §8.8 long-horizon CG-EGA EP regression.

    Architecture follows :class:`HybridCNNGRU`: multi-kernel CNN over
    the 17 dynamic features, two-layer GRU, static patient embedding
    used as a cross-attention query over the GRU output sequence; the
    fused vector feeds a single head that emits ``n_traj_steps`` raw
    mg/dL values (residual learning over Persistence is *not* applied
    here; pure seq2seq prediction).

    At evaluation time the standard 30 / 60 / 90 minute metrics are read
    from indices 5 / 11 / 17 of the trajectory (0-indexed t+5..t+90),
    which align exactly with the existing ``y[:, 0/1/2]`` targets in the
    sequences NPZ (verified to machine precision by
    ``src/build_trajectory_targets.py``).
    """

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        n_traj_steps: int = 18,
        cnn_channels_per_kernel: int = HYBRID_DEFAULTS["cnn_channels_per_kernel"],
        cnn_kernels: tuple[int, ...] = HYBRID_DEFAULTS["cnn_kernels"],
        hidden_dim: int = HYBRID_DEFAULTS["hidden_dim"],
        num_layers: int = HYBRID_DEFAULTS["num_layers"],
        static_embed_dim: int = HYBRID_DEFAULTS["static_embed_dim"],
        attn_dim: int = HYBRID_DEFAULTS["attn_dim"],
        attn_heads: int = HYBRID_DEFAULTS["attn_heads"],
        head_hidden_dim: int = HYBRID_DEFAULTS["head_hidden_dim"],
        dropout: float = HYBRID_DEFAULTS["dropout"],
    ):
        super().__init__()
        self.cnn = _MultiKernelCNN1d(
            n_in=int(n_dynamic),
            n_per_kernel=int(cnn_channels_per_kernel),
            kernels=tuple(cnn_kernels),
            dropout=dropout,
        )
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.static_branch = _StaticBranch(
            int(n_static), embed_dim=int(static_embed_dim), dropout=dropout,
        )
        self.cross_attn = _StaticCrossAttention(
            static_dim=int(static_embed_dim),
            seq_dim=int(hidden_dim),
            attn_dim=int(attn_dim),
            n_heads=int(attn_heads),
            dropout=dropout,
        )
        fuse_dim = int(hidden_dim) + int(attn_dim) + int(static_embed_dim)
        self.head = _MultiHorizonHead(
            in_dim=fuse_dim,
            hidden_dim=int(head_hidden_dim),
            n_horizons=int(n_traj_steps),
            dropout=dropout,
        )
        self.n_traj_steps = int(n_traj_steps)
        self.config = dict(
            model_type="hybrid_cnn_gru_seq2seq",
            n_dynamic=int(n_dynamic),
            n_static=int(n_static),
            n_traj_steps=int(n_traj_steps),
            cnn_channels_per_kernel=int(cnn_channels_per_kernel),
            cnn_kernels=tuple(int(k) for k in cnn_kernels),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            static_embed_dim=int(static_embed_dim),
            attn_dim=int(attn_dim),
            attn_heads=int(attn_heads),
            head_hidden_dim=int(head_hidden_dim),
            dropout=float(dropout),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x_dyn)
        gru_out, _ = self.gru(cnn_out)
        last_h = gru_out[:, -1, :]
        stat_emb = self.static_branch(x_stat)
        attended = self.cross_attn(stat_emb, gru_out)
        fused = torch.cat([last_h, attended, stat_emb], dim=-1)
        return self.head(fused)  # (B, n_traj_steps)


class HybridCNNGRUPersResid(HybridCNNGRU):
    """Step 6 v2 ablation: persistence residual learning.

    The base ``HybridCNNGRU`` outputs three delta values relative to the
    last raw glucose value in the lookback window. The final prediction
    is ``last_glucose_mgdl + delta``. The last raw glucose is computed
    by un-scaling ``x_dyn[:, -1, glucose_idx]`` with the per-subject
    scaler statistics passed to the constructor.

    Why this might help long-horizon CG-EGA: by construction the model
    inherits Persistence's predicted *value* at the limit of zero delta,
    so the predicted trajectory's rate-of-change defaults to "no change"
    rather than to an arbitrary learned constant. The model only needs
    to learn corrections, which is a strictly easier task than learning
    the full prediction from scratch.

    Parameters
    ----------
    pid_glucose_mean, pid_glucose_std
        Two ``(n_patients,)`` numeric tensors (registered as buffers) of
        the per-subject glucose mean and std loaded from
        ``outputs/models/scalers.json``. Used to invert the z-score on
        the last glucose value at forward time.
    pid_lookup
        Mapping from participant ID (string) to row index in the
        ``pid_glucose_*`` tensors. The dataset must supply per-sample
        pid indices via ``x_stat[:, -1]`` (the last static-feature
        column reserved for this purpose).
    glucose_dyn_idx
        Index of the raw ``glucose`` column in ``x_dyn``.
    """

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        pid_glucose_mean: torch.Tensor,
        pid_glucose_std: torch.Tensor,
        glucose_dyn_idx: int,
        **kwargs,
    ):
        # The pid index column is appended to x_static by the runner, so
        # the base hybrid sees one fewer static feature than what is
        # actually passed at forward time. We pass n_static unchanged and
        # slice the pid column off before the static MLP.
        super().__init__(n_dynamic=n_dynamic, n_static=n_static, **kwargs)
        self.register_buffer("pid_glucose_mean", pid_glucose_mean.float())
        self.register_buffer("pid_glucose_std", pid_glucose_std.float())
        self.glucose_dyn_idx = int(glucose_dyn_idx)
        self.config["model_type"] = "hybrid_cnn_gru_pers_resid"
        self.config["glucose_dyn_idx"] = int(glucose_dyn_idx)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        # Strip the pid index from x_stat and use it to look up per-subject
        # glucose scaler stats.
        pid_idx = x_stat[:, -1].long().clamp_min(0)
        x_stat_real = x_stat[:, :-1]
        mean = self.pid_glucose_mean[pid_idx]    # (B,)
        std = self.pid_glucose_std[pid_idx]      # (B,)
        last_glu_z = x_dyn[:, -1, self.glucose_dyn_idx]  # (B,)
        last_glu_mgdl = last_glu_z * std + mean           # (B,) in mg/dL
        delta = super().forward(x_dyn, x_stat_real)       # (B, H)
        return delta + last_glu_mgdl.unsqueeze(1)         # broadcast to (B, H)

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

"""
Model architectures.

  MortalityGRU -- Track 2 from the design doc: sequence model over hourly
                  vitals/labs with an explicit missingness mask. This is the
                  "gap #1" lever: current FL+MIMIC-IV papers all use aggregated
                  tabular features, which caps the achievable ceiling.

  MortalityMLP -- Track 1 baseline on min/max/mean/count summary features,
                  matching what the existing literature does. Keep this so you
                  have an apples-to-apples comparison against published numbers.

Both expose .body_keys() / .head_keys() so FedPer-style personalization can keep
the classifier head local per client while federating the shared representation.
"""
import torch
import torch.nn as nn


class MortalityGRU(nn.Module):
    def __init__(self, n_features, n_static, hidden=128, dropout=0.3, layers=1,
                 channels=2):
        super().__init__()
        # channels=2 -> [value, mask]; channels=3 -> [value, mask, delta]
        self.gru = nn.GRU(
            input_size=channels * n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, mask, static):
        h = torch.cat([x, mask], dim=-1)          # (B, T, 2F)
        out, _ = self.gru(h)
        z = self.drop(self.norm(out[:, -1, :]))   # last hidden state
        return self.head(torch.cat([z, static], dim=-1)).squeeze(-1)

    def body_keys(self):
        return [k for k in self.state_dict() if not k.startswith("head.")]

    def head_keys(self):
        return [k for k in self.state_dict() if k.startswith("head.")]


class MortalityMLP(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x, mask=None, static=None):
        return self.head(self.body(x)).squeeze(-1)

    def body_keys(self):
        return [k for k in self.state_dict() if k.startswith("body.")]

    def head_keys(self):
        return [k for k in self.state_dict() if k.startswith("head.")]


class MortalityHybrid(nn.Module):
    """GRU over the time series, fused with an MLP over aggregate features.

    Rationale: the two views capture different things. The GRU sees trajectory
    (is the lactate rising?), while the tabular branch sees window-level extremes
    and measurement counts (worst lactate; how many gases were drawn) that a
    24-step average tends to wash out. Fusing them consistently beats either
    alone on clinical time series, and it costs almost nothing to compute.
    """

    def __init__(self, n_features, n_tab, hidden=128, dropout=0.3,
                 layers=1, bidirectional=True, channels=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=channels * n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        seq_dim = hidden * (2 if bidirectional else 1)
        self.seq_norm = nn.LayerNorm(seq_dim)

        self.tab = nn.Sequential(
            nn.Linear(n_tab, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(seq_dim + 64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, mask, static):
        # mask carries [observed, delta] when USE_DELTA is on, so this is
        # F + 2F = 3F channels; otherwise F + F = 2F.
        out, _ = self.gru(torch.cat([x, mask], dim=-1))
        # mean-pool AND last state: pooling stabilises when a client is small
        z = self.seq_norm(out.mean(dim=1) + out[:, -1, :])
        return self.head(torch.cat([self.drop(z), self.tab(static)], dim=-1)).squeeze(-1)

    def body_keys(self):
        return [k for k in self.state_dict() if not k.startswith("head.")]

    def head_keys(self):
        return [k for k in self.state_dict() if k.startswith("head.")]


def build_model(cfg, n_features, n_static):
    if cfg.MODEL == "hybrid":
        ch = 3 if getattr(cfg, "USE_DELTA", False) else 2
        return MortalityHybrid(n_features, n_static, cfg.HIDDEN, cfg.DROPOUT,
                               bidirectional=getattr(cfg, "BIDIRECTIONAL", True),
                               channels=ch)
    if cfg.MODEL == "gru":
        ch = 3 if getattr(cfg, "USE_DELTA", False) else 2
        return MortalityGRU(n_features, n_static, cfg.HIDDEN, cfg.DROPOUT,
                            channels=ch)
    if cfg.MODEL == "mlp":
        return MortalityMLP(n_features, cfg.HIDDEN, cfg.DROPOUT)
    raise ValueError(f"unknown MODEL: {cfg.MODEL}")

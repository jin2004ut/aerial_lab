# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class NarxMeta:
    lags: int
    delay: int
    feature_names_single_slice: tuple[str, ...]
    hidden: int
    dropout: float
    dt_est: float
    mu: torch.Tensor
    std: torch.Tensor

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names_single_slice)

    @property
    def input_dim(self) -> int:
        return self.lags * self.feature_dim

    @classmethod
    def from_file(cls, meta_path: str | Path, device: torch.device | str) -> "NarxMeta":
        meta_raw = json.loads(Path(meta_path).read_text())
        return cls(
            lags=int(meta_raw["lags"]),
            delay=int(meta_raw["delay"]),
            feature_names_single_slice=tuple(meta_raw["feature_names_single_slice"]),
            hidden=int(meta_raw["hidden"]),
            dropout=float(meta_raw["dropout"]),
            dt_est=float(meta_raw.get("dt_est", 0.0)),
            mu=torch.tensor(meta_raw["mu"], dtype=torch.float32, device=device),
            std=torch.tensor(meta_raw["std"], dtype=torch.float32, device=device),
        )


class NarxMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def load_narx_model(
    model_path: str | Path,
    meta_path: str | Path,
    device: torch.device | str,
) -> tuple[NarxMLP, NarxMeta]:
    meta = NarxMeta.from_file(meta_path, device=device)
    model = NarxMLP(input_dim=meta.input_dim, hidden_dim=meta.hidden, dropout=meta.dropout).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, meta

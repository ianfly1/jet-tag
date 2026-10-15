from __future__ import annotations

import torch
import torch.nn as nn


class JetMLP(nn.Module):

    def __init__(
        self,
        input_size: int = 90,
        hidden_dims: list[int] | None = None,
        output_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = input_size

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, output_size)

    def forward(
        self, x: torch.Tensor, return_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim > 2:
            x = x.flatten(start_dim=1)

        latent = self.feature_extractor(x)
        out = self.classifier(latent)

        if return_latent:
            return out, latent
        return out
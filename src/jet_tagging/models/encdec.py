from __future__ import annotations

import torch
import torch.nn as nn


class encdec(nn.Module):

    def __init__(
        self,
        input_size: int = 90,
        latent_dim: int = 16,
        output_size: int = 2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, output_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:

        (reconstructed_input, classification_logits)
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        logits = self.classifier(latent)

        return reconstructed, logits
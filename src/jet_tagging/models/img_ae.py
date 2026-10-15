from __future__ import annotations

import torch
import torch.nn as nn


class ImageAutoencoder(nn.Module):

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(128 * 7 * 7, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, 128 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=0
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                128, 1, kernel_size=3, stride=2, padding=1, output_padding=0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.bottleneck(x)

        x = self.decoder_linear(latent)
        x = x.view(-1, 128, 7, 7)
        reconstructed = self.decoder(x)
        return reconstructed
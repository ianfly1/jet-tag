from __future__ import annotations

import torch
import torch.nn as nn


class JetCNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        im_size: int = 25,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dynamically calculate flattened spatial dimension
        dummy_input = torch.zeros(1, in_channels, im_size, im_size)
        conv_out_dim = self.conv_layers(dummy_input).view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        logits = self.fc_layers(x)
        return logits
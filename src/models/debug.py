"""Modelo mínimo para pruebas de humo del ciclo de entrenamiento. No se usa en resultados."""

import torch
from torch import nn


class DebugCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=128, stride=32), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(wav.unsqueeze(1))

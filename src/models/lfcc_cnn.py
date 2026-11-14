"""Modelo A: LFCC + LCNN ligera (Lavrentyeva et al. / baseline ASVspoof).

Entrada: forma de onda [B, 64600]. Los LFCC se calculan dentro del modelo (contrato de datos).
Salida: logits [B, 2], columna 1 = sintética.

Arquitectura: Max-Feature-Map (MFM) + convoluciones 2D + pooling, sin LSTM, para que quepa
en Kaggle T4/P100 y sea exportable a ONNX con facilidad.
"""

from __future__ import annotations

import torch
from torch import nn

from src.features.lfcc import LFCCFrontend


class MaxFeatureMap2D(nn.Module):
    """MFM: max entre la primera y la segunda mitad de canales (Wu et al., Light CNN)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


class LFCCCNN(nn.Module):
    def __init__(self, cfg: dict | None = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        audio = cfg.get("audio", {})
        feats = {**cfg.get("features", {}), **kwargs}
        sample_rate = int(audio.get("sample_rate", 16000))
        self.frontend = LFCCFrontend(
            sample_rate=sample_rate,
            n_lfcc=int(feats.get("n_lfcc", 60)),
            win_ms=int(feats.get("win_ms", 20)),
            hop_ms=int(feats.get("hop_ms", 10)),
            n_fft=int(feats.get("n_fft", 512)),
            deltas=bool(feats.get("deltas", True)),
        )
        # LCNN ligera: canales de entrada = 1 (mapa LFCC), salida de MFM = mitad.
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.frontend(wav)
        x = self.backbone(x)
        return self.head(x)

    def lfcc_map(self, wav: torch.Tensor) -> torch.Tensor:
        """Mapa LFCC [B, 1, frames, coeff] para Grad-CAM / explicabilidad."""
        return self.frontend(wav)

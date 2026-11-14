"""Extracción de LFCC (+ Δ + ΔΔ) dentro del modelo A.

El espectrograma se calcula con Conv1d (bases cos/sen) para poder exportar a ONNX.
`torch.stft` / `unfold` no son fiables en el exportador TorchScript.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def linear_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Banco triangular en Hz lineal → [n_fft//2+1, n_filters]."""
    n_freqs = n_fft // 2 + 1
    points = torch.linspace(0, sample_rate / 2, n_filters + 2)
    bins = torch.floor((n_fft + 1) * points / sample_rate).long()
    fb = torch.zeros(n_freqs, n_filters)
    for i in range(n_filters):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_freqs - 1)
        for j in range(left, center):
            if j < n_freqs:
                fb[j, i] = (j - left) / (center - left)
        for j in range(center, right):
            if j < n_freqs:
                fb[j, i] = (right - j) / (right - center)
    return fb


def dct_matrix(n_lfcc: int, n_filters: int) -> torch.Tensor:
    """DCT-II ortonormal [n_filters, n_lfcc]."""
    n = torch.arange(n_filters, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(n_lfcc, dtype=torch.float32).unsqueeze(0)
    mat = torch.cos(math.pi / n_filters * (n + 0.5) * k)
    mat[:, 0] *= 1.0 / math.sqrt(2.0)
    mat *= math.sqrt(2.0 / n_filters)
    return mat


def _dft_conv_kernels(n_fft: int, win_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Filtros Conv1d que calculan Re/Im del rFFT con ventana de Hann."""
    window = torch.hann_window(win_length)
    if win_length < n_fft:
        window = nn.functional.pad(window, (0, n_fft - win_length))
    n_freqs = n_fft // 2 + 1
    time = torch.arange(n_fft, dtype=torch.float32)
    freqs = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)
    angle = 2.0 * math.pi * freqs * time / n_fft
    # [n_freqs, 1, n_fft]
    cos_k = (window * torch.cos(angle)).unsqueeze(1)
    sin_k = (window * (-torch.sin(angle))).unsqueeze(1)
    return cos_k, sin_k


class LFCCFrontend(nn.Module):
    """Forma de onda [B, T] -> [B, 1, n_frames, n_coeff * (1|3)]."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 60,
        win_ms: int = 20,
        hop_ms: int = 10,
        n_fft: int = 512,
        deltas: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = int(sample_rate * win_ms / 1000)
        self.hop_length = int(sample_rate * hop_ms / 1000)
        self.deltas = deltas
        self.out_dim = n_lfcc * (3 if deltas else 1)

        cos_k, sin_k = _dft_conv_kernels(n_fft, self.win_length)
        # Conv1d fijo: no se entrena.
        self.spec_real = nn.Conv1d(1, n_fft // 2 + 1, kernel_size=n_fft, stride=self.hop_length, bias=False)
        self.spec_imag = nn.Conv1d(1, n_fft // 2 + 1, kernel_size=n_fft, stride=self.hop_length, bias=False)
        with torch.no_grad():
            self.spec_real.weight.copy_(cos_k)
            self.spec_imag.weight.copy_(sin_k)
        self.spec_real.weight.requires_grad_(False)
        self.spec_imag.weight.requires_grad_(False)

        self.register_buffer("filterbank", linear_filterbank(n_lfcc, n_fft, sample_rate))
        self.register_buffer("dct", dct_matrix(n_lfcc, n_lfcc))

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, coeff, frames]. Derivada central sin Conv1d dinámico (ONNX).
        left = x[:, :, :1]
        right = x[:, :, -1:]
        padded = torch.cat([left, x, right], dim=2)
        return 0.5 * (padded[:, :, 2:] - padded[:, :, :-2])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Padding central equivalente a Spectrogram(center=True).
        pad = self.n_fft // 2
        x = nn.functional.pad(wav.unsqueeze(1), (pad, pad))
        real = self.spec_real(x)
        imag = self.spec_imag(x)
        power = real.pow(2) + imag.pow(2)  # [B, n_freq, frames]
        # [B, frames, n_freq] @ filterbank
        filtered = torch.matmul(power.transpose(1, 2), self.filterbank)
        log_fb = torch.log(torch.clamp(filtered, min=1e-10))
        lfcc = torch.matmul(log_fb, self.dct)  # [B, frames, n_lfcc]
        feats = lfcc.transpose(1, 2)  # [B, n_lfcc, frames]
        if self.deltas:
            d1 = self._delta(feats)
            d2 = self._delta(d1)
            feats = torch.cat([feats, d1, d2], dim=1)
        return feats.transpose(1, 2).unsqueeze(1)

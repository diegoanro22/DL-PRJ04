"""Augmentations de audio para train. Contrato en src/data/README.md (sección 5).

Todas comparten la firma `fn(wav: np.ndarray, sr: int) -> np.ndarray`.
`telephony`: simula canal telefónico (8 kHz + pasa-banda 300–3400 Hz).
`rawboost`: RawBoost (Tak et al., 2022) con el algoritmo 3 usado en AASIST.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np
from scipy import signal

Augment = Callable[[np.ndarray, int], np.ndarray]

# Hiperparámetros por defecto de RawBoost (AASIST / Tak et al. 2022), algoritmo 3.
RAWBOOST_DEFAULTS = {
    "algo": 3,
    "nBands": 5,
    "minF": 20,
    "maxF": 8000,
    "minBW": 100,
    "maxBW": 1000,
    "minCoeff": 10,
    "maxCoeff": 100,
    "minG": 0,
    "maxG": 0,
    "minBiasLinNonLin": 5,
    "maxBiasLinNonLin": 20,
    "N_f": 5,
    "P": 10,
    "g_sd": 2.0,
    "SNRmin": 10,
    "SNRmax": 40,
}


def telephony(wav: np.ndarray, sr: int) -> np.ndarray:
    """Baja a 8 kHz y vuelve a `sr`, y aplica un pasa-banda de 300 a 3400 Hz."""
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        return wav.astype(np.float32)
    # resample_poly es mucho más rápido que signal.resample (FFT) en clips largos.
    from math import gcd
    g = gcd(int(sr), 8000)
    up_8k, down_8k = 8000 // g, int(sr) // g
    narrow = signal.resample_poly(wav, up_8k, down_8k)
    wide = signal.resample_poly(narrow, down_8k, up_8k)
    if wide.size != wav.size:
        # Ajuste fino por redondeo de polyphase.
        if wide.size > wav.size:
            wide = wide[: wav.size]
        else:
            wide = np.pad(wide, (0, wav.size - wide.size))
    # Butterworth de orden 4 en banda telefónica.
    low, high = 300 / (sr / 2), 3400 / (sr / 2)
    high = min(high, 0.99)
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.lfilter(b, a, wide)
    return _safe_float32(filtered)


def rawboost(wav: np.ndarray, sr: int, **overrides) -> np.ndarray:
    """RawBoost (Tak et al., 2022): ruido convolutivo, impulsivo y aditivo coloreado."""
    cfg = {**RAWBOOST_DEFAULTS, **overrides}
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.astype(np.float32)
    algo = int(cfg["algo"])
    # Algo 3 (AASIST): LnL + ISD + SSI en serie.
    if algo in (1, 3):
        x = _lnl_convolutive_noise(
            x, cfg["N_f"], cfg["nBands"], cfg["minF"], cfg["maxF"], cfg["minBW"],
            cfg["maxBW"], cfg["minCoeff"], cfg["maxCoeff"], cfg["minG"], cfg["maxG"],
            cfg["minBiasLinNonLin"], cfg["maxBiasLinNonLin"], sr,
        )
    if algo in (2, 3):
        x = _isd_additive_noise(x, cfg["P"], cfg["g_sd"])
    if algo in (3, 4, 5):
        x = _ssi_additive_noise(
            x, cfg["SNRmin"], cfg["SNRmax"], cfg["nBands"], cfg["minF"], cfg["maxF"],
            cfg["minBW"], cfg["maxBW"], cfg["minCoeff"], cfg["maxCoeff"],
            cfg["minG"], cfg["maxG"], sr,
        )
    return _safe_float32(x)


# --- RawBoost helpers (adaptados de TakHemlata/RawBoost-antispoofing, MIT) -------------------

def _rand_range(lo: float, hi: float, integer: bool = False):
    y = float(np.random.uniform(lo, hi))
    return int(y) if integer else y


def _norm_wav(x: np.ndarray, always: bool) -> np.ndarray:
    peak = np.amax(np.abs(x))
    if peak > 0 and (always or peak > 1):
        x = x / peak
    return x


def _gen_notch_coeffs(
    n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs,
):
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = _rand_range(min_f, max_f)
        bw = _rand_range(min_bw, max_bw)
        c = _rand_range(min_coeff, max_coeff, integer=True)
        if c % 2 == 0:
            c += 1
        f1 = max(fc - bw / 2, 1 / 1000)
        f2 = min(fc + bw / 2, fs / 2 - 1 / 1000)
        if f1 >= f2:
            continue
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)
        gain = _rand_range(min_g, max_g)
        _, h = signal.freqz(b, 1, fs=fs)
        peak = np.amax(np.abs(h))
        if peak > 0:
            b = (10 ** (gain / 20)) * b / peak
    return b


def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[0] + 1
    xpad = np.pad(x, (0, n))
    y = signal.lfilter(b, 1, xpad)
    return y[n // 2 : y.shape[0] - n // 2]


def _lnl_convolutive_noise(
    x, n_f, n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff,
    min_g, max_g, min_bias, max_bias, fs,
):
    y = np.zeros_like(x)
    g_lo, g_hi = min_g, max_g
    for i in range(n_f):
        if i == 1:
            g_lo = min_g - min_bias
            g_hi = max_g - max_bias
        b = _gen_notch_coeffs(
            n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, g_lo, g_hi, fs,
        )
        y = y + _filter_fir(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm_wav(y, False)


def _isd_additive_noise(x: np.ndarray, p_pct: float, g_sd: float) -> np.ndarray:
    beta = _rand_range(0, p_pct)
    y = deepcopy(x)
    n = int(x.shape[0] * (beta / 100))
    if n <= 0:
        return _norm_wav(y, False)
    idx = np.random.permutation(x.shape[0])[:n]
    f_r = ((2 * np.random.rand(n)) - 1) * ((2 * np.random.rand(n)) - 1)
    y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _norm_wav(y, False)


def _ssi_additive_noise(
    x, snr_min, snr_max, n_bands, min_f, max_f, min_bw, max_bw,
    min_coeff, max_coeff, min_g, max_g, fs,
):
    noise = np.random.normal(0, 1, x.shape[0])
    b = _gen_notch_coeffs(
        n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs,
    )
    noise = _norm_wav(_filter_fir(noise, b), True)
    snr = _rand_range(snr_min, snr_max)
    denom = np.linalg.norm(noise, 2)
    if denom > 0:
        noise = noise / denom * np.linalg.norm(x, 2) / (10.0 ** (0.05 * snr))
        x = x + noise
    return x


def _safe_float32(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.amax(np.abs(x))
    if peak > 1.0:
        x = x / peak
    return x.astype(np.float32)

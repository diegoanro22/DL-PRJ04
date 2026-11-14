"""Pruebas de augmentations telefónicas y RawBoost."""

import numpy as np
import pytest

from src.data import augment


@pytest.fixture
def tone():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def test_telephony_misma_longitud_y_finito(tone):
    wav, sr = tone
    out = augment.telephony(wav, sr)
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # El pasa-banda atenúa un tono de 440 Hz poco; la energía no debe ser cero.
    assert np.abs(out).mean() > 0


def test_telephony_rechaza_dc_fuera_de_banda():
    sr = 16000
    # Señal casi DC: el pasa-banda 300–3400 la anula.
    wav = np.ones(sr, dtype=np.float32) * 0.5
    out = augment.telephony(wav, sr)
    assert np.abs(out).mean() < 0.05


def test_rawboost_misma_longitud(tone):
    wav, sr = tone
    np.random.seed(0)
    out = augment.rawboost(wav, sr)
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_rawboost_cambia_la_senal(tone):
    wav, sr = tone
    np.random.seed(1)
    out = augment.rawboost(wav, sr)
    assert not np.allclose(out, wav, atol=1e-3)

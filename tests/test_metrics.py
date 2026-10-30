import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    bootstrap_ci,
    compute_auc,
    compute_eer,
    compute_min_tdcf,
    eer_by_attack,
)


def test_eer_separacion_perfecta():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1])
    eer, threshold = compute_eer(scores, labels)
    assert eer == 0.0
    assert 0.3 < threshold <= 0.8


def test_eer_invertido_es_100():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([0, 0, 1, 1])
    assert compute_eer(scores, labels)[0] == 1.0


def test_eer_caso_a_mano():
    # 4 genuinas y 4 sintéticas con un solapamiento: 1 de 4 errores en cada lado -> EER 25 %.
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.85, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert compute_eer(scores, labels)[0] == pytest.approx(0.25)


def test_eer_aleatorio_cerca_de_50():
    rng = np.random.default_rng(0)
    scores = rng.random(20_000)
    labels = rng.integers(0, 2, 20_000)
    assert compute_eer(scores, labels)[0] == pytest.approx(0.5, abs=0.02)


def test_una_sola_clase_da_nan():
    assert np.isnan(compute_eer(np.array([0.1, 0.2]), np.array([1, 1]))[0])
    assert np.isnan(compute_auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_bootstrap_contiene_el_valor():
    rng = np.random.default_rng(1)
    labels = np.r_[np.zeros(500), np.ones(500)].astype(int)
    scores = np.r_[rng.normal(0, 1, 500), rng.normal(2, 1, 500)]
    eer = compute_eer(scores, labels)[0]
    lo, hi = bootstrap_ci(scores, labels, lambda s, y: compute_eer(s, y)[0], n=200)
    assert lo <= eer <= hi
    assert hi - lo < 0.1


def test_min_tdcf_cm_perfecto_es_cero():
    # Con un CM perfecto, el mínimo de t-DCF se alcanza sin errores del CM: 0.
    rng = np.random.default_rng(2)
    asv_keys = np.array(["target"] * 200 + ["nontarget"] * 200 + ["spoof"] * 200)
    asv_scores = np.r_[rng.normal(3, 1, 200), rng.normal(-3, 1, 200), rng.normal(2, 1, 200)]
    scores = np.r_[np.full(50, 0.05), np.full(50, 0.95)]
    labels = np.r_[np.zeros(50), np.ones(50)].astype(int)
    assert compute_min_tdcf(scores, labels, asv_keys, asv_scores) == pytest.approx(0.0, abs=1e-9)


def test_min_tdcf_cm_aleatorio_cerca_de_1():
    # Un CM sin información no mejora a "aceptar todo" o "rechazar todo": t-DCF normalizado ~1.
    rng = np.random.default_rng(3)
    asv_keys = np.array(["target"] * 500 + ["nontarget"] * 500 + ["spoof"] * 500)
    asv_scores = np.r_[rng.normal(3, 1, 500), rng.normal(-3, 1, 500), rng.normal(2, 1, 500)]
    scores = rng.random(4000)
    labels = rng.integers(0, 2, 4000)
    assert compute_min_tdcf(scores, labels, asv_keys, asv_scores) == pytest.approx(1.0, abs=0.05)


def test_eer_por_ataque():
    df = pd.DataFrame({
        "score": [0.1, 0.2, 0.9, 0.95, 0.15],
        "label": [0, 0, 1, 1, 1],
        "attack": ["-", "-", "A01", "A01", "A02"],
    })
    out = eer_by_attack(df).set_index("grupo")
    assert out.loc["A01", "eer"] == 0.0      # A01 se separa perfecto
    assert out.loc["A02", "eer"] > 0.0       # A02 queda entre las genuinas
    assert out.loc["A01", "n_genuinas"] == 2

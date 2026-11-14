"""Pruebas del modelo de valor contra casos calculados a mano (PLAN §12)."""

import numpy as np
import pandas as pd
import pytest

from src.value.value_model import (
    DEFAULT_PARAMS,
    evaluate_model,
    rates_at_threshold,
    value_at_thresholds,
)


def test_detector_perfecto():
    # TPR=1, FPR=0 en ambos umbrales.
    params = {**DEFAULT_PARAMS, "N": 1000, "p": 0.01, "L": 1000.0,
              "e_v": 0.7, "e_a": 0.95, "c_v": 2.0, "c_a": 10.0, "c_comp": 0.0}
    out = value_at_thresholds(1.0, 1.0, 0.0, 0.0, params)
    # fraude_evitado = 1000*0.01*1000*(0.95*1 + 0.7*0) = 9500
    # costo_alertas = 0 (genuinas) + 1000*0.01*(0 + 10*1) = 100
    assert out["fraude_evitado"] == pytest.approx(9500)
    assert out["costo_alertas"] == pytest.approx(100)
    assert out["valor_usd_mes"] == pytest.approx(9400)


def test_detector_aleatorio_umbral_05():
    # Score aleatorio ~ U(0,1): en t=0.5, TPR≈FPR≈0.5.
    params = {**DEFAULT_PARAMS, "N": 10_000, "p": 0.01, "L": 1000.0,
              "e_v": 1.0, "e_a": 1.0, "c_v": 1.0, "c_a": 1.0, "c_comp": 0.0}
    out = value_at_thresholds(0.5, 0.5, 0.5, 0.5, params)
    # fraude = 10000*0.01*1000*(1*0.5) = 50000
    # costo = 10000*0.99*(1*0.5) + 10000*0.01*(1*0.5) = 4950 + 50 = 5000
    assert out["fraude_evitado"] == pytest.approx(50_000)
    assert out["costo_alertas"] == pytest.approx(5_000)
    assert out["valor_usd_mes"] == pytest.approx(45_000)


def test_rates_at_threshold():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    tpr, fpr = rates_at_threshold(scores, labels, 0.5)
    assert tpr == pytest.approx(1.0)
    assert fpr == pytest.approx(0.0)


def test_grid_search_elige_umbral_util(tmp_path):
    # Scores perfectamente separados: el óptimo debe tener FPR bajo y TPR alto.
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0, 0.3, 200), rng.uniform(0.7, 1.0, 200)])
    labels = np.array([0] * 200 + [1] * 200)
    path = tmp_path / "scores.csv"
    pd.DataFrame({"utt_id": range(400), "score": scores, "label": labels}).to_csv(path, index=False)
    params = {**DEFAULT_PARAMS, "c_comp": 0.0}
    result, curve = evaluate_model(path, "perfecto", params, n_grid=21)
    assert result.valor_usd_mes > 0
    assert result.fpr_t1 < 0.2
    assert result.tpr_t2 > 0.8
    assert not curve.empty

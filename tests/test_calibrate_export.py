"""Pruebas de la calibración de Platt y de la exportación a ONNX."""

import numpy as np
import pytest

from src import calibrate
from src.export_onnx import export, load_model
from src.metrics import compute_eer


def _scores_sobreconfiados(rng, n=4000):
    """Scores con la escala del modelo distorsionada: el llr real es logit(score) / 3."""
    labels = rng.integers(0, 2, n)
    true_llr = rng.normal(np.where(labels == 1, 2.0, -2.0), 2.0)
    return calibrate.sigmoid(3 * true_llr), labels, true_llr


def test_platt_recupera_la_escala():
    scores, labels, _ = _scores_sobreconfiados(np.random.default_rng(0))
    params = calibrate.fit(scores, labels)
    # Para dos gaussianas con varianza 4 y medias ±2, el llr exacto es x; el score es 3x.
    assert params["a"] == pytest.approx(1 / 3, rel=0.1)
    assert params["b"] == pytest.approx(0.0, abs=0.1)


def test_calibracion_mejora_ece_y_no_cambia_el_eer():
    scores, labels, _ = _scores_sobreconfiados(np.random.default_rng(1))
    params = calibrate.fit(scores, labels)
    w = calibrate.balanced_weights(labels)
    cal = calibrate.calibrated_probability(scores, params)
    assert calibrate.ece(cal, labels, w) < calibrate.ece(scores, labels, w)
    assert compute_eer(cal, labels)[0] == pytest.approx(compute_eer(scores, labels)[0])


def test_prevalencia_baja_baja_la_probabilidad():
    params = {"a": 1.0, "b": 0.0}
    s = np.array([0.5, 0.9])
    p50 = calibrate.calibrated_probability(s, params, prior=0.5)
    p1 = calibrate.calibrated_probability(s, params, prior=0.01)
    assert p50[0] == pytest.approx(0.5)
    assert np.all(p1 < p50)
    # Bayes a mano: llr = logit(0.9) = log 9; odds = 9 · 0.01 / 0.99.
    assert p1[1] == pytest.approx(9 * 0.01 / 0.99 / (1 + 9 * 0.01 / 0.99))


def test_exporta_aasist_l_a_onnx(tmp_path):
    model, _, name = load_model("configs/aasist_l.yaml")
    out = tmp_path / "b.onnx"
    assert export(model, out) < 1e-4
    assert out.exists() and name == "aasist_l"

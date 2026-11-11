"""Pruebas de la API con un AASIST-L sin entrenar exportado a ONNX."""

import io
import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import api
from app.inference import ModelRegistry, decode_audio, fixed_window
from src.export_onnx import export, load_model

SR = 16000


def wav_bytes(seconds: float, amplitude: float = 0.1, sr: int = SR, fmt: str = "WAV") -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    wav = amplitude * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    for name in ("B_prueba", "B_calibrado"):
        (root / name).mkdir()
    model, _, _ = load_model("configs/aasist_l.yaml")
    export(model, root / "B_prueba" / "model.onnx")
    shutil.copy(root / "B_prueba" / "model.onnx", root / "B_calibrado" / "model.onnx")
    (root / "B_calibrado" / "calibration.json").write_text(json.dumps({"a": 1.0, "b": 0.0}))
    (root / "B_calibrado" / "thresholds.json").write_text(
        json.dumps({"t1": 0.0, "t2": 1.0, "tasa_alertas_esperada": 0.01}))
    return root


@pytest.fixture
def client(models_dir, monkeypatch):
    monkeypatch.setattr(api, "registry", ModelRegistry(models_dir, default="B_prueba"))
    monkeypatch.setattr(api, "monitor", api.Monitor())
    return TestClient(api.app)


def test_health_y_modelos(client):
    assert client.get("/health").json() == {
        "estado": "ok", "modelos": ["B_calibrado", "B_prueba"], "modelo_por_defecto": "B_prueba"}
    info = {m["nombre"]: m for m in client.get("/models").json()}
    assert info["B_prueba"]["umbrales_provisionales"] and not info["B_prueba"]["calibrado"]
    assert info["B_calibrado"]["calibrado"]


def test_analiza_wav(client):
    r = client.post("/analyze", files={"audio": ("a.wav", wav_bytes(5), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 1
    assert body["accion"] in {"continuar", "verificar", "analista"}
    assert body["p_sintetica"] is None          # sin calibration.json
    assert body["avisos"] == []
    assert body["umbrales_provisionales"]


def test_umbrales_y_calibracion(client):
    r = client.post("/analyze?modelo=B_calibrado&prevalencia=0.5",
                    files={"audio": ("a.wav", wav_bytes(5), "audio/wav")})
    body = r.json()
    # t1 = 0 y t2 = 1: todo score cae en "verificar".
    assert body["accion"] == "verificar"
    # a = 1, b = 0 y prevalencia 0.5: la probabilidad calibrada es el mismo score.
    assert body["p_sintetica"] == pytest.approx(body["score"], abs=1e-5)


def test_avisos_de_calidad():
    assert any("corto" in w for w in decode_audio(wav_bytes(0.5)).warnings)
    assert any("silencio" in w for w in decode_audio(wav_bytes(5, amplitude=1e-5)).warnings)
    assert any("saturado" in w for w in decode_audio(wav_bytes(5, amplitude=3.0)).warnings)


def test_remuestrea_y_lee_flac():
    audio = decode_audio(wav_bytes(5, sr=44100, fmt="FLAC"))
    assert audio.duration_s == pytest.approx(5, abs=0.01)
    assert fixed_window(audio.wav).shape == (64600,)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requiere ffmpeg")
def test_lee_webm_del_navegador(client):
    webm = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-c:a", "libopus",
         "-f", "webm", "pipe:1"], input=wav_bytes(5), capture_output=True, check=True).stdout
    r = client.post("/analyze", files={"audio": ("a.webm", webm, "audio/webm")})
    assert r.status_code == 200
    assert r.json()["duracion_s"] == pytest.approx(5, abs=0.1)


def test_errores(client):
    assert client.post("/analyze?modelo=X", files={"audio": ("a.wav", wav_bytes(1))}).status_code == 404
    assert client.post("/analyze", files={"audio": ("a.wav", b"no es audio")}).status_code == 422
    assert client.post("/analyze", files={"audio": ("a.wav", b"")}).status_code == 422
    assert client.get("/metrics").json()["errores"] == {"modelo_desconocido": 1, "audio_invalido": 2}


def test_metricas_y_alarma_de_deriva(client):
    for _ in range(3):
        client.post("/analyze", files={"audio": ("a.wav", wav_bytes(5))})
    m = client.get("/metrics").json()["modelos"]["B_prueba"]
    assert m["llamadas"] == 3 and sum(m["histograma_scores"]) == 3
    assert m["alarma_deriva"] is False          # pocas llamadas y sin tasa esperada

    # B_calibrado alerta en el 100 % de las llamadas y espera 1 %: con 100 llamadas, alarma.
    for _ in range(100):
        api.monitor.record("B_calibrado", 0.5, "verificar", 10.0)
    client.get("/models")  # carga el modelo para leer su tasa esperada
    assert client.get("/metrics").json()["modelos"]["B_calibrado"]["alarma_deriva"] is True


def test_solo_decodifica_los_primeros_segundos():
    from app.inference import MAX_DECODE_SECONDS
    # WAV largo (lo lee soundfile) y FLAC a 44.1 kHz (lo remuestrea ffmpeg): ambos con tope.
    assert decode_audio(wav_bytes(60)).duration_s == pytest.approx(MAX_DECODE_SECONDS)
    if shutil.which("ffmpeg"):
        flac = decode_audio(wav_bytes(60, sr=44100, fmt="FLAC"))
        assert flac.duration_s == pytest.approx(MAX_DECODE_SECONDS, abs=0.01)


def test_reempaquetar_sin_calibracion_borra_la_vieja(tmp_path, monkeypatch):
    import sys

    from app import package_model
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"a": 1.0, "b": 0.0}))
    base = ["prog", "--checkpoint", "configs/aasist_l.yaml", "--name", "m",
            "--models-dir", str(tmp_path / "models")]
    monkeypatch.setattr(sys, "argv", [*base, "--calibration", str(cal), "--t1", "0.3", "--t2", "0.8"])
    package_model.main()
    assert (tmp_path / "models" / "m" / "calibration.json").exists()

    monkeypatch.setattr(sys, "argv", base)
    package_model.main()
    assert not (tmp_path / "models" / "m" / "calibration.json").exists()
    assert not (tmp_path / "models" / "m" / "thresholds.json").exists()

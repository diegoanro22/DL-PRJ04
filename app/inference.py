"""Inferencia en CPU con ONNX Runtime: audio -> P(sintética) -> acción para el agente.

Cada modelo servido vive en una carpeta `<MODELS_DIR>/<nombre>/` con:
    model.onnx          exportado con `python -m src.export_onnx`
    calibration.json    opcional, de `python -m src.calibrate` (a, b de Platt)
    thresholds.json     opcional: {"t1": ..., "t2": ..., "tasa_alertas_esperada": ...}
                        t1 y t2 son umbrales sobre el score crudo, elegidos por valor.

No depende de PyTorch: solo numpy, soundfile, onnxruntime y ffmpeg (para webm/ogg/mp3).
"""

import io
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600       # 4 s: la ventana con la que se entrenó el modelo
MIN_SECONDS = 1.0         # con menos audio la predicción no es confiable
# Se analizan hasta MAX_DECODE_SECONDS en ventanas de 4 s y la decisión sale del promedio.
# El tope evita que un audio comprimido chico se expanda a horas de PCM y agote la memoria.
MAX_DECODE_SECONDS = 20.0
# Una ventana vota solo si tiene voz: al menos SPEECH_MIN_S en tramos de 25 ms por encima de
# SPEECH_RMS (≈ -46 dBFS). El criterio es absoluto, no relativo a la ventana más fuerte: así un
# ruido fuerte al inicio no puede dejar fuera a un clon que habla más bajo. El silencio no aporta
# evidencia sobre la voz (y es un sesgo conocido de ASVspoof), por eso no vota.
SPEECH_RMS = 0.005
SPEECH_FRAME = 400        # 25 ms a 16 kHz
SPEECH_MIN_S = 0.8
SILENCE_RMS = 1e-3        # aprox. -60 dBFS
CLIP_FRACTION = 0.01      # más de 1 % de muestras saturadas
DEFAULT_THRESHOLDS = {"t1": 0.5, "t2": 0.9, "tasa_alertas_esperada": None}
DEFAULT_PRIOR = 0.002     # prevalencia de ataques: la misma que supone src/value/value_model.py

ACTIONS = {
    "continuar": "Continuar la llamada normalmente.",
    "verificar": "Pedir un factor de verificación adicional antes de operar.",
    "analista": "Transferir la llamada al analista de fraude.",
}


class AudioError(ValueError):
    """Audio que no se puede leer o que no tiene contenido."""


@dataclass
class Audio:
    wav: np.ndarray           # float32 mono a 16 kHz, completo
    duration_s: float
    warnings: list[str] = field(default_factory=list)


def _decode_ffmpeg(data: bytes) -> np.ndarray:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-t", str(MAX_DECODE_SECONDS),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            input=data, capture_output=True, check=True, timeout=30,
        )
    except FileNotFoundError as e:
        raise AudioError("ffmpeg no está instalado: solo se aceptan WAV/FLAC a 16 kHz") from e
    except subprocess.CalledProcessError as e:
        raise AudioError("No se pudo leer el audio (formato no soportado o archivo dañado)") from e
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def decode_audio(data: bytes) -> Audio:
    """Bytes de WAV/FLAC/OGG/WebM/MP3 -> mono 16 kHz, con avisos de calidad.

    Decodifica como máximo los primeros MAX_DECODE_SECONDS; `duration_s` es la duración
    decodificada (tope MAX_DECODE_SECONDS).
    """
    if not data:
        raise AudioError("El archivo de audio está vacío")
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            sr = f.samplerate
            if sr == SAMPLE_RATE:
                wav = f.read(int(MAX_DECODE_SECONDS * sr), dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
            else:
                wav = _decode_ffmpeg(data)
    except (sf.SoundFileError, RuntimeError, TypeError):
        # WebM/Opus del navegador, MP3 y otros formatos que soundfile no lee.
        wav = _decode_ffmpeg(data)

    if wav.size == 0 or not np.isfinite(wav).all():
        raise AudioError("El audio no tiene muestras válidas")

    duration = wav.size / SAMPLE_RATE
    warnings = []
    if duration < MIN_SECONDS:
        warnings.append(f"Audio muy corto ({duration:.1f} s): la predicción es poco confiable.")
    elif duration < NUM_SAMPLES / SAMPLE_RATE:
        warnings.append(f"Audio de {duration:.1f} s: el modelo espera 4 s, se repitió el audio.")
    if np.sqrt(np.mean(wav ** 2)) < SILENCE_RMS:
        warnings.append("El audio está casi en silencio: revisar el micrófono o la línea.")
    if np.mean(np.abs(wav) >= 0.999) > CLIP_FRACTION:
        warnings.append("El audio está saturado (clipping): la predicción puede fallar.")
    return Audio(wav=wav, duration_s=round(duration, 2), warnings=warnings)


def fixed_window(wav: np.ndarray) -> np.ndarray:
    """Primeras 64,600 muestras, repitiendo si es más corto (igual que dev/test)."""
    if wav.size < NUM_SAMPLES:
        wav = np.tile(wav, int(np.ceil(NUM_SAMPLES / wav.size)))
    return np.clip(wav[:NUM_SAMPLES], -1, 1).astype(np.float32)


def has_speech(window: np.ndarray) -> bool:
    """True si la ventana tiene al menos SPEECH_MIN_S de tramos con energía de voz."""
    n = window.size // SPEECH_FRAME
    frames = window[: n * SPEECH_FRAME].reshape(n, SPEECH_FRAME)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return int((rms >= SPEECH_RMS).sum()) * SPEECH_FRAME / SAMPLE_RATE >= SPEECH_MIN_S


def windows(wav: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Ventanas de 4 s sin solapamiento que cubren todo el audio: [(inicio, ventana), ...].

    Un audio de hasta 4 s da una sola ventana (repetida, igual que en dev/test). Si sobra un
    tramo al final, la última ventana se alinea con el final del audio para no perderlo.
    """
    if wav.size <= NUM_SAMPLES:
        return [(0, fixed_window(wav))]
    starts = list(range(0, wav.size - NUM_SAMPLES + 1, NUM_SAMPLES))
    if starts[-1] + NUM_SAMPLES < wav.size:
        starts.append(wav.size - NUM_SAMPLES)
    return [(s, fixed_window(wav[s:s + NUM_SAMPLES])) for s in starts]


def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return float(np.log(p) - np.log1p(-p))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


class ServedModel:
    def __init__(self, folder: Path, threads: int = 2):
        import onnxruntime as ort

        self.name = folder.name
        self.folder = folder
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.calibration = self._read_json("calibration.json")
        thresholds = self._read_json("thresholds.json")
        self.provisional = thresholds is None
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        if not self.thresholds["t1"] <= self.thresholds["t2"]:
            raise ValueError(f"{folder}/thresholds.json: se requiere t1 <= t2")

    def _read_json(self, name: str) -> dict | None:
        path = self.folder / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def score(self, window: np.ndarray) -> float:
        """P(sintética) cruda del modelo (softmax), la misma escala que results/scores/."""
        return self.scores(window[None, :])[0]

    def scores(self, batch: np.ndarray) -> list[float]:
        """Scores crudos de varias ventanas [N, 64600] en una sola pasada."""
        logits = self.session.run(None, {"wav": batch})[0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs[:, 1].tolist()

    def probability(self, score: float, prior: float) -> float | None:
        """P(sintética) calibrada para una prevalencia dada; None si no hay calibración."""
        if self.calibration is None:
            return None
        llr = self.calibration["a"] * _logit(score) + self.calibration["b"]
        return _sigmoid(llr + _logit(prior))

    def action(self, score: float) -> str:
        if score >= self.thresholds["t2"]:
            return "analista"
        if score >= self.thresholds["t1"]:
            return "verificar"
        return "continuar"

    def info(self) -> dict:
        return {
            "nombre": self.name,
            "umbrales": {"t1": self.thresholds["t1"], "t2": self.thresholds["t2"]},
            "umbrales_provisionales": self.provisional,
            "calibrado": self.calibration is not None,
        }


class ModelRegistry:
    """Carga perezosa de los modelos de MODELS_DIR (cada subcarpeta con model.onnx)."""

    def __init__(self, models_dir: Path, default: str | None = None, threads: int = 2):
        self.models_dir = models_dir
        self.threads = threads
        self._loaded: dict[str, ServedModel] = {}
        available = self.available()
        self.default = default if default in available else (available[0] if available else None)

    def available(self) -> list[str]:
        if not self.models_dir.is_dir():
            return []
        return sorted(p.name for p in self.models_dir.iterdir() if (p / "model.onnx").is_file())

    def get(self, name: str | None) -> ServedModel:
        name = name or self.default
        if name is None:
            raise KeyError(f"No hay modelos en {self.models_dir}")
        if name not in self._loaded:
            if name not in self.available():
                raise KeyError(f"Modelo desconocido: {name}. Disponibles: {self.available()}")
            self._loaded[name] = ServedModel(self.models_dir / name, self.threads)
        return self._loaded[name]


def analyze(model: ServedModel, audio: Audio, prior: float = DEFAULT_PRIOR) -> dict:
    """Score de cada ventana de 4 s y decisión con el promedio de las que tienen voz.

    El promedio se hace en log-odds (la evidencia de cada ventana), no en probabilidad: así una
    ventana saturada en 0.9999 no domina, y el resultado vuelve a la escala del score crudo
    para compararlo con t1/t2 y calibrarlo igual que una sola ventana.
    """
    t0 = time.perf_counter()
    parts = windows(audio.wav)
    raw = model.scores(np.stack([w for _, w in parts]))
    latency_ms = (time.perf_counter() - t0) * 1000

    voiced = np.array([has_speech(w) for _, w in parts])
    if not voiced.any():
        voiced[:] = True
    score = _sigmoid(float(np.mean([_logit(s) for s, v in zip(raw, voiced) if v])))
    action = model.action(score)
    ventanas = [
        {"inicio_s": round(start / SAMPLE_RATE, 2),
         "fin_s": round(min(start + NUM_SAMPLES, audio.wav.size) / SAMPLE_RATE, 2),
         "score": round(s, 6), "usada": bool(v)}
        for (start, _), s, v in zip(parts, raw, voiced)
    ]
    return {
        "modelo": model.name,
        "score": round(score, 6),
        "p_sintetica": None if (p := model.probability(score, prior)) is None else round(p, 6),
        "prevalencia_supuesta": prior,
        "accion": action,
        "accion_texto": ACTIONS[action],
        "umbrales": {"t1": model.thresholds["t1"], "t2": model.thresholds["t2"]},
        "umbrales_provisionales": model.provisional,
        "duracion_s": audio.duration_s,
        "ventanas": ventanas,
        "avisos": audio.warnings,
        "latencia_ms": round(latency_ms, 1),
    }


def registry_from_env() -> ModelRegistry:
    root = Path(__file__).resolve().parent
    return ModelRegistry(
        Path(os.environ.get("MODELS_DIR", root / "models")),
        default=os.environ.get("DEFAULT_MODEL"),
        threads=int(os.environ.get("ONNX_THREADS", "2")),
    )

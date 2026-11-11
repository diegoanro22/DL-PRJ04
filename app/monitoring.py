"""Monitoreo en memoria para GET /metrics: conteos, latencias, scores y alarma de deriva.

No guarda audio ni datos del cliente: solo el score, la acción y la latencia de cada llamada.
La alarma de deriva compara la tasa de alertas reciente con la esperada en el umbral elegido
(thresholds.json): si un sintetizador nuevo engaña al modelo, o el canal cambia, la tasa se
mueve mucho antes de que se note en las pérdidas.
"""

import threading
import time
from collections import Counter, deque

import numpy as np

WINDOW = 500             # últimas llamadas que se usan para la tasa reciente
MIN_FOR_ALARM = 100      # no alarmar con pocas llamadas
DRIFT_FACTOR = 3.0       # alarma si la tasa sale de [esperada / 3, esperada × 3]
SCORE_BINS = np.linspace(0, 1, 11)


class Monitor:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.total = Counter()                       # llamadas por modelo
        self.actions = Counter()                     # (modelo, acción)
        self.errors = Counter()                      # tipo de error
        self.recent: dict[str, deque] = {}           # modelo -> (score, acción, latencia_ms)

    def record(self, model: str, score: float, action: str, latency_ms: float) -> None:
        with self._lock:
            self.total[model] += 1
            self.actions[(model, action)] += 1
            self.recent.setdefault(model, deque(maxlen=WINDOW)).append((score, action, latency_ms))

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] += 1

    def models(self) -> list[str]:
        with self._lock:
            return list(self.recent)

    def snapshot(self, expected_rates: dict[str, float | None]) -> dict:
        with self._lock:
            models = {}
            for name, calls in list(self.recent.items()):
                scores = np.array([c[0] for c in calls])
                latencies = np.array([c[2] for c in calls])
                alert_rate = float(np.mean([c[1] != "continuar" for c in calls]))
                expected = expected_rates.get(name)
                drift = (
                    expected is not None and len(calls) >= MIN_FOR_ALARM and expected > 0
                    and not expected / DRIFT_FACTOR <= alert_rate <= expected * DRIFT_FACTOR
                )
                models[name] = {
                    "llamadas": self.total[name],
                    "acciones": {a: self.actions[(name, a)]
                                 for a in ("continuar", "verificar", "analista")},
                    "recientes": len(calls),
                    "tasa_alertas_reciente": round(alert_rate, 4),
                    "tasa_alertas_esperada": expected,
                    "alarma_deriva": bool(drift),
                    "latencia_ms": {"p50": round(float(np.percentile(latencies, 50)), 1),
                                    "p95": round(float(np.percentile(latencies, 95)), 1)},
                    "histograma_scores": np.histogram(scores, SCORE_BINS)[0].tolist(),
                }
            return {
                "desde": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
                "modelos": models,
                "errores": dict(self.errors),
            }

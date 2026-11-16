"""API del detector de voz clonada para el centro de contacto.

    POST /analyze   audio (wav, flac, webm, ogg, mp3) -> P(sintética) y acción para el agente
    GET  /models    modelos disponibles con sus umbrales
    GET  /health    estado del servicio
    GET  /metrics   conteos, latencias, distribución de scores y alarma de deriva

Uso local:
    uv run uvicorn app.api:app --reload
    curl -F audio=@llamada.wav "localhost:8000/analyze?modelo=C2"

Variables de entorno: MODELS_DIR (por defecto app/models), DEFAULT_MODEL, ONNX_THREADS,
MAX_UPLOAD_MB.
"""

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.inference import DEFAULT_PRIOR, AudioError, analyze, decode_audio, registry_from_env
from app.monitoring import Monitor

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

app = FastAPI(
    title="Detector de voz clonada",
    description="Analiza los primeros segundos de una llamada y sugiere una acción al agente.",
    version="0.1.0",
)
registry = registry_from_env()
monitor = Monitor()


@app.get("/health")
def health() -> dict:
    available = registry.available()
    return {"estado": "ok" if available else "sin_modelos", "modelos": available,
            "modelo_por_defecto": registry.default}


@app.get("/models")
def models() -> list[dict]:
    return [registry.get(name).info() for name in registry.available()]


@app.post("/analyze")
async def analyze_audio(
    audio: Annotated[UploadFile, File(description="Audio de la llamada; se analizan hasta 20 s en ventanas de 4 s")],
    modelo: Annotated[str | None, Query(description="Modelo; por defecto DEFAULT_MODEL")] = None,
    prevalencia: Annotated[float, Query(gt=0, lt=1, description="Prevalencia de ataques para "
                                        "calibrar la probabilidad")] = DEFAULT_PRIOR,
) -> dict:
    try:
        model = registry.get(modelo)
    except KeyError as e:
        monitor.record_error("modelo_desconocido")
        raise HTTPException(status_code=404, detail=str(e.args[0])) from e

    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        monitor.record_error("audio_muy_grande")
        raise HTTPException(status_code=413,
                            detail=f"El audio supera {MAX_UPLOAD_BYTES // 1024**2} MB")
    # Decodificar y correr el modelo usa CPU: en un hilo aparte para no bloquear el servidor.
    try:
        decoded = await run_in_threadpool(decode_audio, data)
    except AudioError as e:
        monitor.record_error("audio_invalido")
        raise HTTPException(status_code=422, detail=str(e)) from e

    result = await run_in_threadpool(analyze, model, decoded, prevalencia)
    monitor.record(model.name, result["score"], result["accion"], result["latencia_ms"])
    return result


@app.get("/metrics")
def metrics() -> dict:
    expected = {}
    for name in monitor.models():
        try:
            expected[name] = registry.get(name).thresholds.get("tasa_alertas_esperada")
        except KeyError:
            expected[name] = None
    return monitor.snapshot(expected)


# Panel del agente (app/static). Se monta al final para no tapar las rutas de la API.
if (STATIC_DIR / "index.html").exists():
    @app.get("/", include_in_schema=False)
    def panel() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

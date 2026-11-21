"""Tablas operativas y contraste del ONNX publicado, a partir de predicciones reales.

Se separa el corte binario en t2 de las tres acciones del sistema. Ningún umbral,
peso, etiqueta, caso local o estrato se ajusta a partir de estas tablas.
"""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.inference import ServedModel
from lab10.acquire import CHECKSUMS, REVISION
from lab10.audit import OUT, sha256, write_json


def deployment_verification(verified, waves, settings):
    """Reutiliza la clase de la API con el C4/model.onnx del Hub fijado por hash.

    Es una comprobación local del artefacto publicado sobre las ventanas originales,
    no una consulta al servicio web ni equivalencia probada para toda perturbación.
    """
    folder = Path(settings.checkpoint).resolve().parent
    path = folder / "model.onnx"
    if not path.is_file():
        raise FileNotFoundError("Falta C4/model.onnx: ejecutar python -m lab10.acquire --onnx")
    checksum = sha256(path)
    if settings.model == "C4" and checksum != CHECKSUMS["model.onnx"]:
        raise ValueError("El ONNX no coincide con el artefacto publicado de C4.")
    start = time.perf_counter()
    # Dos hilos: mismo valor predeterminado de ServedModel/ONNX_THREADS en la API.
    served = ServedModel(folder, threads=2)
    load_seconds = time.perf_counter() - start
    records = []
    start = time.perf_counter()
    for row in verified.itertuples():
        score = float(served.score(waves[(row.split, row.utt_id)].numpy()))
        records.append({"utt_id": row.utt_id, "split": row.split, "attack": row.attack,
                        "label": row.label, "gpu_cached_score": row.cached_score,
                        "gpu_cached_action": row.cached_action,
                        "pytorch_cpu_score": row.score, "pytorch_cpu_action": row.action,
                        "onnx_cpu_score": score, "onnx_cpu_action": served.action(score),
                        "abs_diff_onnx_pytorch": abs(score - row.score)})
    frame = pd.DataFrame(records)
    inference_seconds = time.perf_counter() - start
    frame.to_csv(OUT / "deployment_verification.csv", index=False)
    changed = frame[(frame.gpu_cached_action != frame.pytorch_cpu_action) |
                    (frame.pytorch_cpu_action != frame.onnx_cpu_action)].copy()
    changed.to_csv(OUT / "runtime_action_changes.csv", index=False)
    summary = {
        "n": len(frame), "model": settings.model, "hub_revision": REVISION,
        "onnx_path": str(path), "onnx_sha256": checksum,
        "onnx_providers": served.session.get_providers(),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "onnx_input_type": served.session.get_inputs()[0].type,
        "onnx_threads": 2,
        "thresholds": served.thresholds,
        "max_abs_diff_onnx_pytorch": float(frame.abs_diff_onnx_pytorch.max()),
        "pytorch_onnx_action_changes": int((frame.pytorch_cpu_action != frame.onnx_cpu_action).sum()),
        "gpu_cached_onnx_action_changes": int((frame.gpu_cached_action != frame.onnx_cpu_action).sum()),
        "load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "inputs": len(frame), "forwards": len(frame),
        "scope": "Artefacto C4 publicado, clase ServedModel de la API, ventanas originales; "
                 "no servicio web remoto ni llamada multiventana. GPU/bf16 es histórico de CSV.",
    }
    write_json(OUT / "deployment_summary.json", summary)
    return frame, changed, summary


def export_decision_tables(local, global_sample, probes, fidelity_frame, thresholds):
    """Exporta tablas de casos, sondas, muestreo y fidelidad."""
    labels = {
        "acierto_score_extremo": "acierto extremo",
        "error": "FN que continúa (A10)",
        "cercano_t2": "cercano a t2",
        "subgrupo": "A10; sin códec tel.",
        "genuino_distinto": "genuino distinto",
    }
    rows = []
    for i, row in enumerate(local.itertuples(), 1):
        distance = abs(row.score - thresholds["t2"])
        inclusion = labels[row.reason]
        if row.reason == "cercano_t2":
            inclusion += f"\n|p−t2|={distance:.8f}"
        if row.reason == "subgrupo":
            inclusion = f"{row.attack}; " + ("sin códec tel." if row.codec == "none" else row.codec)
        rows.append({"case": i, "utt_id": row.utt_id,
                     "binary_real_to_predicted": f"{row.label}→{row.prediction}",
                     "score": row.score, "score_decimal": f"{row.score:.6f}",
                     "action": row.action, "inclusion": inclusion,
                     "attack": row.attack, "codec": row.codec,
                     "distance_to_t2": distance,
                     "historical_distance_to_t2": abs(row.cached_score - thresholds["t2"]),
                     "synthetic_allowed_to_continue": bool(row.label == 1 and row.action == "continuar"),
                     "synthetic_sent_to_review": bool(row.label == 1 and row.action == "verificar")})
    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / "local_cases_table.csv", index=False)
    case_map = cases.set_index("utt_id")["case"]
    changes = probes[probes.action != probes.base_action].copy()
    changes["case"] = changes.utt_id.map(case_map)
    changes.to_csv(OUT / "probe_action_changes.csv", index=False)
    directions = probes.groupby(["probe", "base_action", "action"]).size().rename("n").reset_index()
    directions.to_csv(OUT / "probe_directions.csv", index=False)
    strata = ["split", "label", "attack", "codec"]
    sampling = {
        "n": len(global_sample),
        "n_genuine": int((global_sample.label == 0).sum()),
        "n_synthetic": int((global_sample.label == 1).sum()),
        "n_attacks": int(global_sample.loc[global_sample.label == 1, "attack"].nunique()),
        "seed": 42, "strata": strata,
        "selection": "Hasta dos por estrato split×clase×ataque×códec con RNG(42), "
                     "sin reemplazo, fijados antes de leer audio; ver global_sample.csv.",
        "scope": "Comparación entre clases exploratoria; dos genuinos no estiman dispersión "
                 "poblacional de esa clase ni prevalencia de llamadas bancarias.",
    }
    write_json(OUT / "sampling_summary.json", sampling)
    top = fidelity_frame[fidelity_frame.regions == "top"].pivot(
        index="utt_id", columns="replacement", values="directional_drop").reset_index()
    top["absolute_ramp_change"] = np.abs(top.hard - top.ramp10ms)
    top["case"] = top.utt_id.map(case_map)
    top.to_csv(OUT / "fidelity_control_summary.csv", index=False)
    return cases, changes, sampling, top

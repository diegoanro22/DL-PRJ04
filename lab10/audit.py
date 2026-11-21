"""Auditoría de evidencia versionada; no genera atribuciones a partir de scores."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lab10/outputs"
TEST_SPLITS = ("asv19_eval", "asv19_eval_codec", "es_test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def repository_audit() -> tuple[pd.DataFrame, dict]:
    """Compara C2/C3/C4 sin decidir qué versión se defendió oralmente."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(ROOT / "results/runs.csv").set_index("run")
    rows = []
    for run in ("C2", "C3", "C4"):
        for split in TEST_SPLITS:
            file = ROOT / f"results/metrics/{run}__{split}.json"
            m = json.loads(file.read_text())
            rows.append({
                "run": run, "split": split, "n": m["n"], "eer": m["eer"],
                "auc": m["auc"], "epoch": m["epoch"],
                "checkpoint": m["checkpoint"],
                "checkpoint_exists": (ROOT / m["checkpoint"]).is_file(),
                "train_split": runs.loc[run, "train_split"],
                "training_commit": runs.loc[run, "commit"],
                "metrics_sha256": sha256(file),
                "scores_sha256": sha256(ROOT / f"results/scores/{run}__{split}.csv"),
            })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    thresholds = json.loads((ROOT / "results/value/C2_thresholds.json").read_text())
    audit = {
        "status": "AUDITORIA_DOCUMENTAL_NO_ES_EJECUCION_XAI",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "documented_deployment": "C4 publicado en r0sV/signet y README actualizado; Dockerfile/PRODUCT conservan configuración histórica C2.",
        "later_results": "C4 mejora es_test; C2 mejora asv19_eval y asv19_eval_codec.",
        "final_presentation": "No está versionada. El usuario seleccionó expresamente C4.",
        "selected_model": "C4: instrucción explícita del equipo y ficha r0sV/signet (modelo servido).",
        "hub_revision": "fa67b890666c8f92477a27d4155032afbf86b405",
        "spanish_audio_access": "El usuario confirmó que solo están en el equipo de su compañero.",
        "C2_thresholds": {k: thresholds[k] for k in ("t1", "t2")},
        "C2_intermediate_zone": thresholds["t1"] < thresholds["t2"],
        "dataset_unit": "primeras 64600 muestras a 16000 Hz = 4.0375 s",
        "api_unit": "hasta 20 s; promedio de logit(score) de ventanas con voz",
        "api_silence_fallback": "si ninguna tiene voz, votan TODAS (app/inference.py)",
        "positive_class": "1 = sintética; 0 = genuina",
        "score": "softmax[:,1] crudo; no es P calibrada para prevalencia bancaria",
        "occlusion_target": "logits[:,1] - logits[:,0], unidades log-odds",
        "published_maps": "No sustituyen cinco casos nuevos ni prueban checkpoint local.",
        "sources": ["README.md", "PRODUCT.md", "DESIGN.md", "docs/escrito/secciones_diego.md",
                    "results/runs.csv", "src/data/dataset.py", "src/explain/occlusion.py",
                    "app/inference.py", "results/value/C2_thresholds.json"],
    }
    write_json(OUT / "repository_audit.json", audit)
    return comparison, audit


if __name__ == "__main__":
    frame, record = repository_audit()
    print(frame[["run", "split", "n", "eer", "checkpoint_exists"]].to_string(index=False))
    print(json.dumps(record, ensure_ascii=False, indent=2))

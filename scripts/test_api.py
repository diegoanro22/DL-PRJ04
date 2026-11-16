"""Pasa un set de prueba propio por la API y resume cuántos audios clasifica bien.

Estructura esperada (cualquier formato que acepte la API):
    data/pruebas/real/*.wav|webm|...        voces genuinas (grabadas con consentimiento)
    data/pruebas/sintetico/*.wav|...        voces generadas (scripts/generate_synthetic.py)
    data/pruebas/clon/*.mp3|...             clones de una voz real (p. ej. de un servicio web)

Cada subcarpeta de primer nivel es una etiqueta; los archivos se buscan de forma recursiva. Las
que empiezan con "real" son genuinas; todas las demás se tratan como voz generada.
Para una real, acierto = "continuar"; para una sintética, acierto = cualquier alerta
("verificar" o "analista").

Uso (con la API corriendo):
    uv run python -m scripts.test_api --modelo C2
    uv run python -m scripts.test_api --modelo C0 --dir data/pruebas --out results/pruebas_propias.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pandas as pd

from src.config import REPO_ROOT

EXTS = {".wav", ".flac", ".ogg", ".webm", ".mp3", ".m4a"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "pruebas")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--modelo", default=None)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "pruebas_propias.csv")
    args = p.parse_args()

    rows = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        for etiqueta_dir in sorted(d for d in args.dir.iterdir() if d.is_dir()):
            for f in sorted(x for x in etiqueta_dir.rglob("*") if x.suffix.lower() in EXTS):
                params = {"modelo": args.modelo} if args.modelo else {}
                r = client.post("/analyze", params=params, files={"audio": (f.name, f.read_bytes())})
                r.raise_for_status()
                body = r.json()
                sintetico = not etiqueta_dir.name.startswith("real")
                rows.append({
                    "etiqueta": etiqueta_dir.name,
                    "archivo": str(f.relative_to(args.dir)),
                    "modelo": body["modelo"],
                    "score": body["score"],
                    "accion": body["accion"],
                    "acierto": (body["accion"] != "continuar") == sintetico,
                    "duracion_s": body["duracion_s"],
                    "ventanas": len(body.get("ventanas", [])),
                    "avisos": " | ".join(body["avisos"]),
                })
    if not rows:
        raise SystemExit(f"No hay audios en {args.dir}/<etiqueta>/")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df[["etiqueta", "archivo", "score", "accion", "acierto"]].to_string(index=False))
    print()
    resumen = df.groupby("etiqueta").agg(
        audios=("acierto", "size"), aciertos=("acierto", "sum"),
        score_mediana=("score", "median"), score_min=("score", "min"), score_max=("score", "max"))
    resumen["tasa"] = (resumen.aciertos / resumen.audios).map("{:.0%}".format)
    print(resumen.to_string())
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()

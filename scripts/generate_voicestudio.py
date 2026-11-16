"""Clona voces de FLEURS con OmniVoice (2025) a través de la API local de VoiceStudio.

VoiceStudio (app de escritorio, AGPL) expone una API en localhost; este script le manda cada
referencia de FLEURS con su transcripción y guarda el audio generado. OmniVoice tiene licencia
CC-BY-NC: los audios sirven para investigación, no para un producto comercial.

Requiere VoiceStudio abierto. Se genera sin efectos ("raw") y la marca de agua debe estar
apagada (se verifica antes de empezar), para no medir la marca en lugar de la voz.

Uso:
    uv run python -m scripts.generate_voicestudio --split train --n 1500 --out data/es_mod/omni/train
    uv run python -m scripts.generate_voicestudio --url http://localhost:3900 --split test --n 300 ...
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import httpx
import soundfile as sf

from src.config import REPO_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:3900")
    p.add_argument("--fleurs-dir", type=Path, default=REPO_ROOT / "data" / "fleurs_es")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with httpx.Client(base_url=args.url, timeout=300) as client:
        marca = client.get("/watermark/status").json()
        if marca.get("invisible_enabled") or marca.get("visible_audio_enabled"):
            raise SystemExit("Apague la marca de agua de audio en VoiceStudio antes de generar")

        with (args.fleurs_dir / f"{args.split}.tsv").open(encoding="utf-8") as f:
            filas = [r for r in csv.reader(f, delimiter="\t") if len(r) > 2]
        rng = random.Random(args.seed)
        rng.shuffle(filas)
        args.out.mkdir(parents=True, exist_ok=True)

        hechos, t0 = 0, time.time()
        for i, ref_fila in enumerate(filas * 2):  # dos vueltas si hacen falta más que filas
            if hechos >= args.n:
                break
            destino = args.out / f"omni_{Path(ref_fila[1]).stem}_{i:05d}.wav"
            if destino.exists():
                hechos += 1
                continue
            ref = args.fleurs_dir / args.split / ref_fila[1]
            if not ref.exists() or not 2 <= sf.info(ref).duration <= 15:
                continue
            texto = filas[(i + 1) % len(filas)][2]
            for intento in range(3):
                try:
                    r = client.post("/generate", data={
                        "text": texto, "language": "es", "ref_text": ref_fila[2],
                        "effect_preset": "raw", "seed": str(args.seed + i)},
                        files={"ref_audio": (ref.name, ref.read_bytes(), "audio/wav")})
                    r.raise_for_status()
                    if not r.content.startswith(b"RIFF"):
                        raise ValueError(f"respuesta inesperada: {r.content[:120]!r}")
                    destino.write_bytes(r.content)
                    hechos += 1
                    break
                except Exception as e:  # noqa: BLE001 - un audio que falla no detiene el lote
                    print(f"intento {intento + 1} falló ({ref.name}): {e}", flush=True)
                    time.sleep(5)
            if hechos and hechos % 100 == 0:
                print(f"  {hechos}/{args.n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"{hechos} audios en {args.out}")


if __name__ == "__main__":
    main()

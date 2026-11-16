# /// script
# requires-python = ">=3.11"
# dependencies = ["edge-tts>=7", "soundfile", "numpy"]
# ///
"""Genera voz sintética moderna en español con Edge TTS (voces neuronales de Microsoft).

Edge TTS es un servicio en línea gratuito: se le manda solo el texto, no audio de nadie. Sirve
como sintetizador comercial actual (distinto de los de ASVspoof 2019 y de XTTS).

Modos:
- Frases del set de prueba (por defecto): cada voz dice las frases de scripts/grabar_reales.py.
- --tsv de FLEURS: genera N textos de ese split repartidos entre las voces (datos de C3).

Uso:
    uv run scripts/generate_edge_tts.py --voces es-GT-AndresNeural es-GT-MartaNeural --out data/pruebas/edge_tts
    uv run scripts/generate_edge_tts.py --tsv data/fleurs_es/train.tsv --n 1500 --voces-entrenamiento \\
        --out data/es_mod/edge/train
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import subprocess
from pathlib import Path

FRASES = [
    "Buenas tardes, quiero hacer una transferencia a otra cuenta.",
    "Hola, llamo porque necesito reponer mi tarjeta de débito.",
    "Mi número de documento es dos cuatro siete uno, cinco ocho tres nueve.",
    "Quisiera cambiar el número de teléfono registrado en mi cuenta.",
    "Necesito autorizar un pago de cinco mil quetzales, es urgente.",
    "Sí, soy el titular de la cuenta, pueden verificarlo.",
    "Perdí mi celular y necesito bloquear la banca en línea.",
    "Quiero aumentar el límite de mi tarjeta de crédito.",
    "Por favor, envíen el dinero a la cuenta que les voy a dictar.",
    "Buenos días, ¿me pueden confirmar el saldo de mi cuenta de ahorro?",
]
# Voces de prueba: Centroamérica y México (el acento de la demo). Nunca se usan para entrenar.
VOCES_PRUEBA = [
    "es-GT-AndresNeural", "es-GT-MartaNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural",
    "es-SV-LorenaNeural", "es-SV-RodrigoNeural", "es-HN-CarlosNeural", "es-HN-KarlaNeural",
]


async def todas_las_voces() -> list[str]:
    import edge_tts
    return sorted(v["ShortName"] for v in await edge_tts.list_voices() if v["Locale"].startswith("es-"))


async def sintetizar(texto: str, voz: str, destino: Path, sem: asyncio.Semaphore) -> bool:
    import edge_tts
    async with sem:
        mp3 = destino.with_suffix(".mp3")
        for intento in range(3):
            try:
                await edge_tts.Communicate(texto, voz).save(str(mp3))
                break
            except Exception as e:  # noqa: BLE001 - errores de red del servicio
                if intento == 2:
                    print(f"falló {destino.name}: {e}", flush=True)
                    return False
                await asyncio.sleep(2)
        # WAV 16 kHz mono, como el resto de los datos (el códec MP3 queda en la señal).
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3),
                        "-ac", "1", "-ar", "16000", str(destino)], check=True)
        mp3.unlink()
        return True


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--voces", nargs="*", help="voces a usar (por defecto, las de prueba)")
    p.add_argument("--voces-entrenamiento", action="store_true",
                   help="todas las voces en español menos las de prueba")
    p.add_argument("--tsv", type=Path, help="TSV de FLEURS (columna 3 = transcripción)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--concurrencia", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.voces_entrenamiento:
        voces = [v for v in await todas_las_voces() if v not in VOCES_PRUEBA]
    else:
        voces = args.voces or VOCES_PRUEBA
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    trabajos: list[tuple[str, str, Path]] = []
    if args.tsv:
        with args.tsv.open(encoding="utf-8") as f:
            textos = list(dict.fromkeys(row[2] for row in csv.reader(f, delimiter="\t") if len(row) > 2))
        rng.shuffle(textos)
        for i, texto in enumerate(textos[: args.n]):
            voz = voces[i % len(voces)]
            trabajos.append((texto, voz, args.out / f"edge_{voz}_{i:05d}.wav"))
    else:
        for voz in voces:
            for i, texto in enumerate(FRASES):
                trabajos.append((texto, voz, args.out / f"edge_{voz}_{i:02d}.wav"))

    sem = asyncio.Semaphore(args.concurrencia)
    pendientes = [t for t in trabajos if not t[2].exists()]
    ok = sum(await asyncio.gather(*(sintetizar(t, v, d, sem) for t, v, d in pendientes)))
    print(f"{ok}/{len(pendientes)} audios nuevos en {args.out} ({len(voces)} voces)")


if __name__ == "__main__":
    asyncio.run(main())

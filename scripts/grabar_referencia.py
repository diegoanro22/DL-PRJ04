"""Graba una muestra de voz de referencia para un servicio de clonación y la guarda en MP3.

Muestra un texto en español con variedad de sonidos, graba (por defecto 45 s) con el micrófono
por defecto y guarda WAV y MP3. Al final imprime las frases del set de prueba para que el clon
diga exactamente lo mismo que las grabaciones reales (scripts/grabar_reales.py).

Solo con la voz propia o con consentimiento de quien habla. Subir la voz a un servicio externo
es compartir un dato biométrico: revisar qué hace ese servicio con el audio.

Uso:
    uv run python -m scripts.grabar_referencia --nombre diego
    uv run python -m scripts.grabar_referencia --nombre diego --segundos 60
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

from scripts.grabar_reales import FRASES
from src.config import REPO_ROOT

TEXTO = """\
Hola, buenas tardes. Mi nombre es el que aparece en la cuenta y llamo desde la ciudad.
Ayer revisé mis movimientos y vi un cobro que no reconozco, por eso quería consultarlo.
También necesito actualizar mi dirección: ahora vivo en la zona diez, cerca del parque.
¿Podrían decirme cuánto tarda en llegar una tarjeta nueva? La última vez fueron tres días.
Muchas gracias por la ayuda; quedo pendiente de su confirmación por correo o por mensaje.
Uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve y diez."""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nombre", required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "propios")
    p.add_argument("--segundos", type=int, default=45)
    p.add_argument("--pausa", type=float, default=5.0, help="segundos para prepararse")
    args = p.parse_args()
    for tool in ("arecord", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Falta {tool}")

    nombre = re.sub(r"[^a-z0-9]+", "_", args.nombre.lower())
    args.out.mkdir(parents=True, exist_ok=True)
    wav = args.out / f"{nombre}_referencia.wav"
    mp3 = wav.with_suffix(".mp3")

    print("Lea el texto en voz alta, con su voz normal y sin prisa. Si termina antes, vuelva a",
          f"empezar: graba {args.segundos} s.\n")
    print("\n".join(f"    {linea}" for linea in TEXTO.splitlines()), "\n", flush=True)
    for s in range(int(args.pausa), 0, -1):
        print(f"    empieza en {s}…", flush=True)
        time.sleep(1)
    print(f"    ● grabando {args.segundos} s…", flush=True)
    # 44.1 kHz: los servicios de clonación piden buena calidad; el detector remuestrea a 16 kHz.
    subprocess.run(["arecord", "-q", "-f", "S16_LE", "-r", "44100", "-c", "1",
                    "-d", str(args.segundos), str(wav)], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                    "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)], check=True)
    print(f"\nListo:\n    {mp3}\n    {wav}\n")
    print("Frases para que las diga el clon (las mismas de las grabaciones reales):\n")
    for i, frase in enumerate(FRASES):
        print(f"    {i:02d}. {frase}")
    print(f"\nGuarde lo que genere el servicio en data/pruebas/clon/ como {nombre}_clon_00.mp3, "
          f"{nombre}_clon_01.mp3, … y corra: uv run python -m scripts.test_api --modelo C2")


if __name__ == "__main__":
    main()

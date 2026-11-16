"""Graba voces reales para el set de prueba propio, sin interacción: muestra cada frase y graba.

Usa las mismas frases que scripts/generate_synthetic.py, así real y sintético dicen lo mismo.
Graba a 16 kHz mono con el micrófono por defecto (arecord). Solo con consentimiento de quien
habla (ver data/consentimiento_plantilla.md); los audios quedan en data/, que no va a git.

Uso:
    uv run python -m scripts.grabar_reales --nombre diego
    uv run python -m scripts.grabar_reales --nombre cisco --segundos 8 --frases 5
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

from src.config import REPO_ROOT

# Mismas frases que scripts/generate_synthetic.py (ese script corre en otro entorno).
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nombre", required=True, help="quién habla (para los archivos)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "pruebas" / "real")
    p.add_argument("--segundos", type=int, default=7)
    p.add_argument("--frases", type=int, default=len(FRASES))
    p.add_argument("--pausa", type=float, default=3.0, help="segundos para leer la frase antes de grabar")
    args = p.parse_args()
    if shutil.which("arecord") is None:
        raise SystemExit("Falta arecord (alsa-utils)")

    nombre = re.sub(r"[^a-z0-9]+", "_", args.nombre.lower())
    args.out.mkdir(parents=True, exist_ok=True)
    for i, frase in enumerate(FRASES[: args.frases]):
        print(f"\n[{i + 1}/{args.frases}] Lea en voz alta, con su voz normal:\n\n    {frase}\n", flush=True)
        time.sleep(args.pausa)
        print(f"    ● grabando {args.segundos} s…", flush=True)
        path = args.out / f"{nombre}_{i:02d}.wav"
        subprocess.run(["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1",
                        "-d", str(args.segundos), str(path)], check=True)
        print(f"    listo: {path.name}", flush=True)
    print(f"\nGrabaciones en {args.out}")


if __name__ == "__main__":
    main()

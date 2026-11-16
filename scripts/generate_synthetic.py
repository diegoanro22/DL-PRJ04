# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "coqui-tts>=0.26,<0.27",
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "transformers>=4.43,<4.50",
#   "soundfile",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchaudio = { index = "pytorch-cu128" }
# ///
"""Genera audio sintético en español con XTTS-v2 para probar el detector fuera de ASVspoof.

Corre en su propio entorno (torch 2.8): con torch >= 2.9 coqui-tts exige torchcodec, que no
soporta el FFmpeg 9 del sistema. XTTS-v2 tiene licencia CPML (solo uso no comercial).

Dos modos:
- Voces incluidas en XTTS (no clona a nadie): --speakers "Ana Florence" ...
- Clon de una persona que dio su consentimiento: --reference su_voz.wav --name diego

Uso:
    COQUI_TOS_AGREED=1 uv run scripts/generate_synthetic.py --n-speakers 6
    COQUI_TOS_AGREED=1 uv run scripts/generate_synthetic.py --reference data/propios/diego_ref.wav --name diego
"""

from __future__ import annotations

import argparse
import re
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


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/pruebas/sintetico"))
    p.add_argument("--speakers", nargs="*", help="voces incluidas de XTTS")
    p.add_argument("--n-speakers", type=int, default=6, help="si no se dan --speakers")
    p.add_argument("--reference", type=Path, help="wav de referencia para clonar (con consentimiento)")
    p.add_argument("--name", default="clon", help="nombre de la persona clonada (para los archivos)")
    p.add_argument("--frases", type=int, default=len(FRASES))
    args = p.parse_args()

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    frases = FRASES[: args.frases]

    if args.reference:
        voces = [(args.name, {"speaker_wav": str(args.reference)})]
    else:
        nombres = args.speakers or tts.speakers[:: max(1, len(tts.speakers) // args.n_speakers)][: args.n_speakers]
        voces = [(n, {"speaker": n}) for n in nombres]

    for nombre, kwargs in voces:
        for i, frase in enumerate(frases):
            path = args.out / f"xtts_{slug(nombre)}_{i:02d}.wav"
            tts.tts_to_file(text=frase, language="es", file_path=str(path), **kwargs)
        print(f"{nombre}: {len(frases)} audios", flush=True)
    print(f"Listo: {args.out}")


if __name__ == "__main__":
    main()

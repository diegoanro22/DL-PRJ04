# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "f5-tts>=1.1,<1.2",
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "soundfile",
#   "huggingface_hub",
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
"""Clona voces en español con F5-TTS (2024, flow matching), afinado en español (jpgallegoar/F5-Spanish).

Es un clonador de la misma generación que los servicios comerciales: con unos segundos de una voz
y su transcripción, dice cualquier texto con esa voz. Corre en su propio entorno (torch 2.8).

Modos:
- Una persona (con consentimiento): --ref su_voz.wav [--ref-text "lo que dice"] -> frases de prueba.
- FLEURS (datos de C3): --fleurs-dir data/fleurs_es --split train --n 600 -> cada clon usa un audio
  de FLEURS como referencia y dice el texto de otra fila.

Uso:
    uv run scripts/generate_f5.py --ref data/propios/diego_referencia.wav --name diego --out data/pruebas/clon_f5
    uv run scripts/generate_f5.py --fleurs-dir data/fleurs_es --split train --n 600 --out data/es_mod/f5/train
"""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
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


def cargar_modelo(device: str):
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download("jpgallegoar/F5-Spanish", "model_1250000.safetensors")
    vocab = hf_hub_download("jpgallegoar/F5-Spanish", "vocab.txt")
    return F5TTS(model="F5TTS_Base", ckpt_file=ckpt, vocab_file=vocab, device=device)


def recortar(ref: Path, inicio: float, segundos: float) -> Path:
    """F5 funciona mejor con referencias de menos de ~12 s que coincidan con --ref-text."""
    import soundfile as sf

    wav, sr = sf.read(ref, dtype="float32", always_2d=True)
    out = Path(tempfile.mkdtemp()) / "ref.wav"
    sf.write(out, wav[int(inicio * sr): int((inicio + segundos) * sr)].mean(axis=1), sr)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref", type=Path)
    p.add_argument("--ref-text", default="", help="transcripción de la referencia (vacío: la transcribe Whisper)")
    p.add_argument("--ref-inicio", type=float, default=0.0, help="segundo donde empieza el tramo de referencia")
    p.add_argument("--ref-segundos", type=float, default=10.0)
    p.add_argument("--name", default="clon")
    p.add_argument("--fleurs-dir", type=Path)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = cargar_modelo(device)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.ref:
        ref = recortar(args.ref, args.ref_inicio, args.ref_segundos)
        for i, frase in enumerate(FRASES):
            tts.infer(ref_file=str(ref), ref_text=args.ref_text, gen_text=frase,
                      file_wave=str(args.out / f"f5_{args.name}_{i:02d}.wav"), seed=args.seed + i)
        print(f"{len(FRASES)} audios en {args.out}")
        return

    # FLEURS: columnas id, archivo, transcripción cruda, ...
    with (args.fleurs_dir / f"{args.split}.tsv").open(encoding="utf-8") as f:
        filas = [r for r in csv.reader(f, delimiter="\t") if len(r) > 2]
    rng = random.Random(args.seed)
    rng.shuffle(filas)
    # Referencias de 2 a 10 s (F5 corta o falla con referencias muy largas).
    import soundfile as sf

    hechos = 0
    for i, ref_fila in enumerate(filas):
        if hechos >= args.n:
            break
        destino = args.out / f"f5_{Path(ref_fila[1]).stem}_{i:05d}.wav"
        if destino.exists():
            hechos += 1
            continue
        ref = args.fleurs_dir / args.split / ref_fila[1]
        if not ref.exists() or not 2 <= sf.info(ref).duration <= 10:
            continue
        texto = filas[(i + 1) % len(filas)][2]
        try:
            tts.infer(ref_file=str(ref), ref_text=ref_fila[2], gen_text=texto,
                      file_wave=str(destino), seed=args.seed + i, show_info=lambda *_: None)
            hechos += 1
        except Exception as e:  # noqa: BLE001 - un audio que falla no detiene el lote
            print(f"falló {ref.name}: {e}", flush=True)
        if hechos % 50 == 0:
            print(f"  {hechos}/{args.n}", flush=True)
    print(f"{hechos} audios en {args.out}")


if __name__ == "__main__":
    main()

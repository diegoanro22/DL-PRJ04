"""Precalcula versiones con códec telefónico de un subconjunto de train.

Los códecs (G.711 μ-law, GSM, AMR-NB) no se aplican al vuelo (ffmpeg es lento).
Este script escribe FLAC en data/asv19_train_codec/ y un protocolo
protocols/asv19_train_codec.csv listo para usar como split de train.

Uso:
    uv run python scripts/precompute_codecs.py \\
        --root data/ASVspoof2019_LA --protocol protocols/asv19_train.csv \\
        --fraction 0.25 --codecs g711 gsm amr
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.config import REPO_ROOT

CODEC_FFMPEG = {
    # μ-law PCM en WAV contenedor.
    "g711": ["-c:a", "pcm_mulaw", "-ar", "8000"],
    "gsm": ["-c:a", "libgsm", "-ar", "8000"],
    "amr": ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1"],
}


def apply_codec(wav: np.ndarray, sr: int, codec: str) -> np.ndarray:
    if codec not in CODEC_FFMPEG:
        raise ValueError(f"Códec desconocido: {codec}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está en PATH")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, enc, dst = tmp / "in.wav", tmp / "enc", tmp / "out.wav"
        sf.write(src, wav, sr)
        enc_args = CODEC_FFMPEG[codec]
        # Codifica a formato estrecho y vuelve a WAV PCM 16 kHz.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), *enc_args, str(enc)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(enc), "-ar", str(sr), "-ac", "1",
             "-c:a", "pcm_s16le", str(dst)],
            check=True, capture_output=True,
        )
        out, _ = sf.read(dst, dtype="float32")
    return np.asarray(out, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precalcula códecs telefónicos sobre train.")
    p.add_argument("--root", type=Path, required=True, help="dataset_roots.asv19")
    p.add_argument("--protocol", type=Path, default=REPO_ROOT / "protocols" / "asv19_train.csv")
    p.add_argument("--out-audio", type=Path, default=REPO_ROOT / "data" / "asv19_train_codec")
    p.add_argument("--out-protocol", type=Path, default=REPO_ROOT / "protocols" / "asv19_train_codec.csv")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--codecs", nargs="+", default=["g711", "gsm", "amr"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.protocol, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(args.seed)
    n = max(1, int(len(df) * args.fraction))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    args.out_audio.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in sample.iterrows():
        wav, sr = sf.read(args.root / row.path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        codec = str(rng.choice(args.codecs))
        try:
            out = apply_codec(wav, sr, codec)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"omitido {row.utt_id} ({codec}): {exc}")
            continue
        rel = f"{row.utt_id}_{codec}.flac"
        sf.write(args.out_audio / rel, out, sr)
        rows.append({
            "utt_id": f"{row.utt_id}_{codec}",
            "dataset": "asv19_codec",
            "path": rel,
            "label": row.label,
            "attack": row.attack,
            "speaker": row.speaker,
            "language": row.language,
            "codec": codec,
        })
    out_df = pd.DataFrame(rows)
    args.out_protocol.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_protocol, index=False)
    print(f"{len(out_df)} audios → {args.out_audio}")
    print(f"Protocolo → {args.out_protocol}")
    print("Añade a configs/machines/*.yaml:")
    print(f"  asv19_codec: {args.out_audio}")


if __name__ == "__main__":
    main()

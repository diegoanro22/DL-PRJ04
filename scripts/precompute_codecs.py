"""Precalcula versiones con códec telefónico de un subconjunto de un split.

Los códecs (G.711 μ-law, GSM, AMR-NB) no se aplican al vuelo (ffmpeg es lento).
Por defecto escribe FLAC en data/asv19_train_codec/ y un protocolo
protocols/asv19_train_codec.csv listo para usar como split de train. Con --protocol de eval
sirve para medir la robustez al canal telefónico (cada audio con un códec al azar).

Uso:
    uv run python -m scripts.precompute_codecs \\
        --root data/ASVspoof2019_LA --protocol protocols/asv19_train.csv \\
        --fraction 0.25 --codecs g711 gsm amr
    uv run python -m scripts.precompute_codecs \\
        --root data/ASVspoof2019_LA --protocol protocols/asv19_eval.csv --fraction 0.2 \\
        --out-audio data/asv19_eval_codec --out-protocol protocols/asv19_eval_codec.csv \\
        --dataset-key asv19_eval_codec --workers 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.config import REPO_ROOT

CODEC_FFMPEG = {
    # μ-law PCM en WAV contenedor.
    "g711": ["-c:a", "pcm_mulaw", "-ar", "8000"],
    "gsm": ["-c:a", "libgsm", "-ar", "8000", "-ac", "1"],
    "amr": ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1"],
}
# ffmpeg elige el contenedor por la extensión; sin ella no puede escribir el archivo.
CODEC_EXT = {"g711": ".wav", "gsm": ".gsm", "amr": ".amr"}


def apply_codec(wav: np.ndarray, sr: int, codec: str) -> np.ndarray:
    if codec not in CODEC_FFMPEG:
        raise ValueError(f"Códec desconocido: {codec}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está en PATH")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, enc, dst = tmp / "in.wav", tmp / f"enc{CODEC_EXT[codec]}", tmp / "out.wav"
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
    p.add_argument("--dataset-key", default="asv19_codec",
                   help="clave de dataset_roots que apunta a --out-audio")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return p.parse_args()


def _process(job: tuple) -> dict | None:
    row, root, out_audio, codec, key = job
    wav, sr = sf.read(root / row["path"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    try:
        out = apply_codec(wav, sr, codec)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"omitido {row['utt_id']} ({codec}): {exc}", flush=True)
        return None
    rel = f"{row['utt_id']}_{codec}.flac"
    sf.write(out_audio / rel, out, sr)
    return {**row, "utt_id": f"{row['utt_id']}_{codec}", "dataset": key, "path": rel,
            "codec": codec}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.protocol, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(args.seed)
    n = max(1, int(len(df) * args.fraction))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    args.out_audio.mkdir(parents=True, exist_ok=True)
    codecs = rng.choice(args.codecs, size=len(sample))
    jobs = [(row, args.root, args.out_audio, str(c), args.dataset_key)
            for row, c in zip(sample.to_dict("records"), codecs)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_process, jobs, chunksize=32), 1):
            if r is not None:
                rows.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    out_df = pd.DataFrame(rows)
    args.out_protocol.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_protocol, index=False)
    print(f"{len(out_df)} audios → {args.out_audio}")
    print(f"Protocolo → {args.out_protocol}")
    print("Añade a configs/machines/*.yaml:")
    print(f"  {args.dataset_key}: {args.out_audio}")


if __name__ == "__main__":
    main()

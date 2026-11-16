"""Protocolos en español con voz real (FLEURS) y sintética moderna (Edge, F5, OmniVoice) para C3/C4.

Fuentes (rutas relativas a dataset_roots.es = data/):
    fleurs_es/<split>/*.wav           real, español latinoamericano (Google FLEURS)
    es_mod/fleurs_mp3/<split>/*.wav   real pasado por MP3 (lo genera este script)
    es_mod/edge/<split>/*.wav         Edge TTS (voces de prueba solo en test)
    es_mod/f5/<split>/*.wav           clones F5-TTS de hablantes de FLEURS
    es_mod/omni/<split>/*.wav         clones OmniVoice (VoiceStudio) de hablantes de FLEURS

El audio de Edge TTS llega en MP3 de 24 kHz; una fracción de los reales se pasa por el mismo
códec para que el modelo no aprenda el atajo "MP3 = sintético".

Escribe protocols/es_{train,dev,test}.csv y, combinados con ASVspoof 2019,
protocols/mix_train.csv y protocols/mix_dev.csv.

Uso:
    uv run python -m scripts.build_es_protocol --mp3-fraction 0.5
"""

from __future__ import annotations

import argparse
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT

DATA = REPO_ROOT / "data"
PROTOCOLS = REPO_ROOT / "protocols"
FIELDS = ["utt_id", "dataset", "path", "label", "attack", "speaker", "language", "codec"]


def mp3_roundtrip(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f'ffmpeg -hide_banner -loglevel error -y -i "{src}" -ar 24000 -b:a 48k -f mp3 - | '
        f'ffmpeg -hide_banner -loglevel error -y -i - -ac 1 -ar 16000 "{dst}"',
        shell=True, check=True)


def rows_for(split: str, mp3_fraction: float, seed: int) -> list[dict]:
    rows = []
    reales = sorted((DATA / "fleurs_es" / split).glob("*.wav"))
    for f in reales:
        rows.append({"utt_id": f"fleurs_{split}_{f.stem}", "path": f"fleurs_es/{split}/{f.name}",
                     "label": 0, "attack": "-", "speaker": "-", "codec": "none"})
    rng = random.Random(seed)
    elegidos = rng.sample(reales, int(len(reales) * mp3_fraction))
    destino = DATA / "es_mod" / "fleurs_mp3" / split
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda f: mp3_roundtrip(f, destino / f.name), elegidos))
    for f in elegidos:
        rows.append({"utt_id": f"fleurs_mp3_{split}_{f.stem}", "path": f"es_mod/fleurs_mp3/{split}/{f.name}",
                     "label": 0, "attack": "-", "speaker": "-", "codec": "mp3"})
    for gen in ("edge", "f5", "omni"):
        for f in sorted((DATA / "es_mod" / gen / split).glob("*.wav")):
            speaker = f.stem.split("_")[1] if gen == "edge" else "-"
            rows.append({"utt_id": f"{gen}_{split}_{f.stem}", "path": f"es_mod/{gen}/{split}/{f.name}",
                         "label": 1, "attack": gen, "speaker": speaker,
                         "codec": "mp3" if gen == "edge" else "none"})
    for r in rows:
        r.update(dataset="es", language="es")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mp3-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for split in ("train", "dev", "test"):
        df = pd.DataFrame(rows_for(split, args.mp3_fraction, args.seed))[FIELDS]
        df.to_csv(PROTOCOLS / f"es_{split}.csv", index=False)
        print(f"es_{split}: {len(df)} audios, por fuente:",
              df.groupby(["attack", "codec"]).size().to_dict())
    for mix, asv, es in (("mix_train", "asv19_train", "es_train"), ("mix_dev", "asv19_dev", "es_dev")):
        df = pd.concat([pd.read_csv(PROTOCOLS / f"{asv}.csv", dtype=str, keep_default_na=False),
                        pd.read_csv(PROTOCOLS / f"{es}.csv", dtype=str, keep_default_na=False)])
        df.to_csv(PROTOCOLS / f"{mix}.csv", index=False)
        print(f"{mix}: {len(df)} audios ({(df.label == '1').mean():.0%} sintéticos)")


if __name__ == "__main__":
    main()

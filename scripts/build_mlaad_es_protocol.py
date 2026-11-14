"""Protocolo del subconjunto en español de MLAAD (+ bonafide de M-AILABS es).

Espera una carpeta con metadatos CSV/JSONL y audio. El layout típico tras preparar
el dataset privado del equipo:

    data/MLAAD_es/
      spoof/   *.wav|flac   (o por modelo TTS)
      bonafide/ *.wav|flac
      meta.csv  (opcional: utt_id,path,label,attack,speaker)

Si existe `meta.csv` con las columnas del contrato, se usa directo. Si no, se
recorren `spoof/` (label=1) y `bonafide/` (label=0).

Uso:
    uv run python scripts/build_mlaad_es_protocol.py --root data/MLAAD_es
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["utt_id", "dataset", "path", "label", "attack", "speaker", "language", "codec"]
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def from_meta(root: Path, meta: Path) -> list[dict]:
    rows = []
    with meta.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            missing = set(FIELDS) - set(row) - {"dataset"}
            if missing:
                raise ValueError(f"{meta}:{i}: faltan columnas {sorted(missing)}")
            path = root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({
                "utt_id": row["utt_id"],
                "dataset": "mlaad_es",
                "path": row["path"],
                "label": int(row["label"]),
                "attack": row.get("attack") or "-",
                "speaker": row.get("speaker") or "-",
                "language": row.get("language") or "es",
                "codec": row.get("codec") or "none",
            })
    return rows


def from_folders(root: Path) -> list[dict]:
    rows = []
    for label, folder, attack in (
        (0, "bonafide", "-"),
        (1, "spoof", "mlaad"),
    ):
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            rel = path.relative_to(root).as_posix()
            utt_id = path.stem
            # Prefijo del subfolder TTS si existe (spoof/xtts/utt.wav → attack=xtts).
            parts = path.relative_to(base).parts
            att = parts[0] if label == 1 and len(parts) > 1 else attack
            rows.append({
                "utt_id": f"{folder}_{utt_id}",
                "dataset": "mlaad_es",
                "path": rel,
                "label": label,
                "attack": att if label == 1 else "-",
                "speaker": "-",
                "language": "es",
                "codec": "none",
            })
    if not rows:
        raise FileNotFoundError(
            f"No hay meta.csv ni carpetas bonafide/spoof con audio en {root}"
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "protocols" / "mlaad_es.csv")
    args = p.parse_args()
    meta = args.root / "meta.csv"
    rows = from_meta(args.root, meta) if meta.is_file() else from_folders(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_spoof = sum(r["label"] for r in rows)
    print(f"{args.output}: {len(rows)} ({len(rows) - n_spoof} genuinos, {n_spoof} sintéticos)")


if __name__ == "__main__":
    main()

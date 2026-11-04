"""Convierte los protocolos oficiales de ASVspoof 2019 LA al contrato CSV del repo.

Uso: uv run python scripts/build_protocols.py --root data/ASVspoof2019_LA
Formato oficial: https://github.com/clovaai/aasist/blob/main/data_utils.py
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["utt_id", "dataset", "path", "label", "attack", "speaker", "language", "codec"]
SPLITS = {"train": "trn", "dev": "trl", "eval": "trl"}


def build_asv19_protocol(root: Path, split: str, check_files: bool = True) -> list[dict]:
    """No lee audio ni scores: solo metadatos y, opcionalmente, existencia de archivos."""
    filename = f"ASVspoof2019.LA.cm.{split}.{SPLITS[split]}.txt"
    protocol = next((p for p in (
        root / "ASVspoof2019_LA_cm_protocols" / filename,
        root / filename,
    ) if p.is_file()), None)
    if protocol is None:
        raise FileNotFoundError(f"No se encontró {filename} en {root} ni en ASVspoof2019_LA_cm_protocols/")
    audio_dir = next((p for p in (
        root / f"ASVspoof2019_LA_{split}" / "flac",
        root / split / "flac",
    ) if p.is_dir()), None)
    if audio_dir is None:
        raise FileNotFoundError(f"No se encontró la carpeta de audio del split {split} en {root}")
    rows, seen = [], set()
    for line_no, line in enumerate(protocol.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[-1] not in ("bonafide", "spoof"):
            raise ValueError(f"{protocol}:{line_no}: se esperaban 5 campos y etiqueta bonafide/spoof")
        speaker, utt_id, _, attack, label = parts
        if Path(utt_id).name != utt_id or utt_id in (".", "..") or utt_id in seen:
            raise ValueError(f"{protocol}:{line_no}: utt_id inválido o duplicado: {utt_id}")
        seen.add(utt_id)
        audio_path = audio_dir / f"{utt_id}.flac"
        if check_files and not audio_path.is_file():
            raise FileNotFoundError(f"{protocol}:{line_no}: falta {audio_path}")
        rows.append({
            "utt_id": utt_id, "dataset": "asv19", "path": audio_path.relative_to(root).as_posix(),
            "label": int(label == "spoof"), "attack": attack if label == "spoof" else "-",
            "speaker": speaker, "language": "en", "codec": "none",
        })
    if not rows:
        raise ValueError(f"{protocol}: protocolo vacío")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Misma raíz que dataset_roots.asv19")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "protocols")
    parser.add_argument("--skip-file-check", action="store_true", help="Omitir comprobación de cada FLAC")
    args = parser.parse_args()
    # Validar todos los splits antes de escribir para no dejar una generación parcial.
    protocols = {split: build_asv19_protocol(args.root, split, not args.skip_file_check)
                 for split in args.splits}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in protocols.items():
        path = args.output_dir / f"asv19_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        n_spoof = sum(row["label"] for row in rows)
        print(f"{path}: {len(rows)} audios ({len(rows) - n_spoof} genuinos, {n_spoof} sintéticos)")


if __name__ == "__main__":
    main()

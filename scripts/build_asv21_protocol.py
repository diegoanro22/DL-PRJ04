"""Protocolo unificado para ASVspoof 2021 LA eval (robustez a canal telefónico).

Espera la carpeta oficial con `ASVspoof2021_LA_eval/flac/` y el trial list
`ASVspoof2021.LA.cm.eval.trl.txt` (o equivalente en la raíz del dataset).

Uso:
    uv run python scripts/build_asv21_protocol.py --root data/ASVspoof2021_LA
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["utt_id", "dataset", "path", "label", "attack", "speaker", "language", "codec"]


def _find_protocol(root: Path) -> Path:
    candidates = [
        root / "ASVspoof2021_LA_cm_protocols" / "ASVspoof2021.LA.cm.eval.trl.txt",
        root / "ASVspoof2021.LA.cm.eval.trl.txt",
        root / "trial_metadata.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No se encontró el protocolo de eval 2021 en {root}. "
        "Esperado ASVspoof2021.LA.cm.eval.trl.txt"
    )


def _find_audio_dir(root: Path) -> Path:
    for path in (
        root / "ASVspoof2021_LA_eval" / "flac",
        root / "flac",
        root / "eval" / "flac",
    ):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No se encontró la carpeta flac de ASVspoof 2021 en {root}")


def build(root: Path, check_files: bool = True) -> list[dict]:
    protocol, audio_dir = _find_protocol(root), _find_audio_dir(root)
    rows, seen = [], set()
    for line_no, line in enumerate(protocol.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        # Formatos vistos: speaker utt - attack label  |  utt label  |  utt attack label codec
        if len(parts) >= 5 and parts[-1] in ("bonafide", "spoof"):
            speaker, utt_id, _, attack, label = parts[:5]
            codec = parts[5] if len(parts) > 5 else "unknown"
        elif len(parts) == 3 and parts[-1] in ("bonafide", "spoof"):
            utt_id, attack, label = parts
            speaker, codec = "-", "unknown"
        else:
            raise ValueError(f"{protocol}:{line_no}: formato no reconocido: {line[:80]}")
        if utt_id in seen:
            raise ValueError(f"{protocol}:{line_no}: utt_id duplicado: {utt_id}")
        seen.add(utt_id)
        # Busca .flac o .wav
        audio_path = next((p for ext in (".flac", ".wav")
                           if (p := audio_dir / f"{utt_id}{ext}").is_file()), None)
        if check_files and audio_path is None:
            raise FileNotFoundError(f"{protocol}:{line_no}: falta audio de {utt_id} en {audio_dir}")
        if audio_path is None:
            audio_path = audio_dir / f"{utt_id}.flac"
        rows.append({
            "utt_id": utt_id,
            "dataset": "asv21",
            "path": audio_path.relative_to(root).as_posix(),
            "label": int(label == "spoof"),
            "attack": attack if label == "spoof" else "-",
            "speaker": speaker,
            "language": "en",
            "codec": codec,
        })
    if not rows:
        raise ValueError(f"{protocol}: protocolo vacío")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "protocols" / "asv21_eval.csv")
    p.add_argument("--skip-file-check", action="store_true")
    args = p.parse_args()
    rows = build(args.root, not args.skip_file_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_spoof = sum(r["label"] for r in rows)
    print(f"{args.output}: {len(rows)} ({len(rows) - n_spoof} genuinos, {n_spoof} sintéticos)")


if __name__ == "__main__":
    main()

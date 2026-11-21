"""Descarga solo los FLAC públicos exactos del manifiesto; sin credenciales.

No regenera el español del proyecto ni reemplaza archivos por disponibilidad.
La API pública de Kaggle devuelve FLAC o ZIP; se valida el archivo antes de copiarlo.
"""
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf

from lab10.audit import OUT, ROOT

DATASET = "awsaf49/asvpoof-2019-dataset"


def download():
    manifest = pd.read_csv(OUT / "required_audio.csv")
    selected = manifest[manifest.dataset == "asv19"].drop_duplicates("path")

    def fetch(item):
        i, row = item
        relative = Path(row.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Ruta inválida en manifest.")
        destination = ROOT / row.audio_path
        if not destination.exists():
            remote = "LA/LA/" + relative.as_posix()
            url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
            for attempt in range(3):
                try:
                    response = requests.get(url, params={"file_name": remote}, timeout=(20, 60))
                    response.raise_for_status()
                    data = response.content
                    if zipfile.is_zipfile(io.BytesIO(data)):
                        with zipfile.ZipFile(io.BytesIO(data)) as archive:
                            matches = [p for p in archive.namelist() if p == remote or p == row.path
                                       or p == relative.name]
                            if len(matches) != 1:
                                raise ValueError(f"Kaggle no devolvió exactamente {remote}")
                            data = archive.read(matches[0])
                    sf.info(io.BytesIO(data))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".pending")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                    break
                except (requests.RequestException, sf.SoundFileError):
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        sf.info(destination)
        print(f"ASVspoof público {i}/{len(selected)}: {row.utt_id}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, enumerate(selected.itertuples(), 1)))
    other = sorted(set(manifest.dataset) - {"asv19"})
    if other:
        print(f"Pendientes datasets fuera de ASVspoof original: {other}")


if __name__ == "__main__":
    download()

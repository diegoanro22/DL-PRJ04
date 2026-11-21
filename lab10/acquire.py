"""Descarga C4 fijado por revisión y hashes; --onnx incluye el artefacto servido.

Los pesos y segmentos reanudables permanecen en runs/ (ignorado por Git).
"""
import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from lab10.audit import ROOT, sha256

REVISION = "fa67b890666c8f92477a27d4155032afbf86b405"
CHECKSUMS = {
    "best.pt": "f83e759c2b4186bd9b7097a81c9c3fb2fde1a6745773e990ee2493ab8139afb0",
    "model.onnx": "a7e03a0410e9685e3cca6a3624910bc9ee9f15929bc1cc8d1a81d0b201c3750b",
    "thresholds.json": "3460dbb4a4c48aedee62b6cd9fe15b87dce819e420b31ab3adfa26f133a561eb",
    "value_thresholds.json": "2dd5ab90664b8b0c379d4c478cca3b3fe2f71b43cb86e7a00ea5f1e1251f1865",
    "calibration.json": "97e0359177e2dafcb497928ae3e03b49f6a6f5364ac433aba48097864b24e158",
}
SIZES = {"best.pt": 1263763238, "model.onnx": 1264265576}
CHUNK = 32 * 1024 * 1024


def download_large(folder, filename):
    """Valida cada rango y finalmente el hash LFS completo, sin usar credenciales."""
    destination = folder / filename
    size = SIZES[filename]
    if not destination.exists():
        parts = folder / f".download-parts-{filename}"
        parts.mkdir(exist_ok=True)
        ranges = [(start, min(start + CHUNK, size) - 1) for start in range(0, size, CHUNK)]
        url = f"https://huggingface.co/r0sV/signet/resolve/{REVISION}/C4/{filename}?download=true"

        def fetch(pair):
            start, end = pair
            dest = parts / f"{start:012}.part"
            if dest.exists() and dest.stat().st_size == end - start + 1:
                return dest
            for attempt in range(4):
                try:
                    with requests.get(url, stream=True, headers={"Range": f"bytes={start}-{end}"},
                                      timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError("El servidor no respetó el rango solicitado.")
                        with dest.open("wb") as target:
                            for data in response.iter_content(1024 * 1024):
                                target.write(data)
                    if dest.stat().st_size != end - start + 1:
                        raise OSError("Descarga incompleta")
                    print(f"C4/{filename}: segmento {start // CHUNK + 1}/{len(ranges)}", flush=True)
                    return dest
                except (requests.RequestException, OSError):
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))

        with ThreadPoolExecutor(max_workers=4) as executor:
            files = list(executor.map(fetch, ranges))
        pending = destination.with_suffix(".pending")
        with pending.open("wb") as target:
            for source in files:
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, target)
        if sha256(pending) != CHECKSUMS[filename]:
            raise ValueError(f"SHA256 incorrecto: {filename}")
        pending.replace(destination)
        shutil.rmtree(parts)
    if sha256(destination) != CHECKSUMS[filename]:
        raise ValueError(f"SHA256 local incorrecto: {filename}")
    print(f"C4/{filename} verificado por SHA256", flush=True)


def acquire(onnx=False):
    folder = ROOT / "runs/C4"
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/r0sV/signet/resolve/{REVISION}/C4/"
    for filename in ("config.yaml", "thresholds.json", "value_thresholds.json", "calibration.json"):
        dest = folder / filename
        if not dest.is_file():
            response = requests.get(base + filename, timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
        if filename in CHECKSUMS and sha256(dest) != CHECKSUMS[filename]:
            raise ValueError(f"Hash incorrecto: {filename}")
    download_large(folder, "best.pt")
    if onnx:
        download_large(folder, "model.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", action="store_true", help="Incluir C4/model.onnx para verificar la API")
    acquire(parser.parse_args().onnx)

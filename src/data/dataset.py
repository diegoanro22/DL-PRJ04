"""Dataset y dataloaders. Contrato completo en src/data/README.md.

Las firmas de este archivo están acordadas: train.py y evaluate.py dependen de ellas.
La implementación está a cargo de Cisco.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600  # 4 s
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1
PROTOCOLS_DIR = Path(__file__).resolve().parents[2] / "protocols"


def protocol_path(split: str) -> Path:
    """Ruta del CSV de un split, p. ej. "asv19_train" -> protocols/asv19_train.csv."""
    return PROTOCOLS_DIR / f"{split}.csv"


def build_dataloader(cfg: dict, split: str, train: bool) -> DataLoader:
    """DataLoader de un split.

    Cada batch es un dict {"wav": [B, 64600] float32, "label": [B] long, "utt_id": list[str]}.
    Con train=True: recorte aleatorio, augmentation según cfg["augment"] y shuffle.
    Con train=False: primeras 64,600 muestras, sin augmentation ni shuffle.
    """
    raise NotImplementedError("Pendiente: Cisco")


def class_weights(split: str) -> torch.Tensor:
    """Pesos [genuina, sintética] inversos a la frecuencia de cada clase en el split."""
    raise NotImplementedError("Pendiente: Cisco")

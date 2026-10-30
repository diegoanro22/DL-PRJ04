"""Dataset sintético SOLO para pruebas de humo de train.py / evaluate.py (--fake-data).

Cumple el mismo contrato que src/data/dataset.py. Las "sintéticas" llevan un tono de 4 kHz
sobre el ruido, así que un modelo que aprende debería bajar el EER casi a 0: sirve para
comprobar que el ciclo de entrenamiento funciona, no para medir nada.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import LABEL_SPOOF, NUM_SAMPLES, SAMPLE_RATE

SIZES = {"train": 512, "dev": 256, "eval": 256}
SPOOF_RATIO = 0.9  # parecido al desbalance de ASVspoof 2019
ATTACKS = ["A01", "A02", "A03", "A04"]
CODECS = ["none", "g711"]


def _size(split: str) -> int:
    for key, size in SIZES.items():
        if key in split:
            return size
    return SIZES["eval"]


def _seed(split: str) -> int:
    return sum(map(ord, split))


def fake_protocol(split: str) -> pd.DataFrame:
    """Protocolo con las mismas columnas que los CSV reales (ver src/data/README.md)."""
    rng = np.random.default_rng(_seed(split))
    n = _size(split)
    labels = (rng.random(n) < SPOOF_RATIO).astype(int)
    return pd.DataFrame({
        "utt_id": [f"FAKE_{split}_{i:05d}" for i in range(n)],
        "dataset": "fake",
        "path": "-",
        "label": labels,
        "attack": [rng.choice(ATTACKS) if y == LABEL_SPOOF else "-" for y in labels],
        "speaker": "-",
        "language": "en",
        "codec": rng.choice(CODECS, size=n),
    })


class FakeDataset(Dataset):
    def __init__(self, split: str, num_samples: int = NUM_SAMPLES):
        self.protocol = fake_protocol(split)
        self.num_samples = num_samples
        self.base_seed = _seed(split)
        t = np.arange(num_samples) / SAMPLE_RATE
        self.tone = (0.1 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.protocol)

    def __getitem__(self, i: int) -> dict:
        row = self.protocol.iloc[i]
        rng = np.random.default_rng(self.base_seed * 100_003 + i)
        wav = (0.1 * rng.standard_normal(self.num_samples)).astype(np.float32)
        if row.label == LABEL_SPOOF:
            wav += self.tone
        return {"wav": torch.from_numpy(wav), "label": int(row.label), "utt_id": row.utt_id}


def build_dataloader(cfg: dict, split: str, train: bool) -> DataLoader:
    ds = FakeDataset(split, cfg["audio"]["num_samples"])
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train, num_workers=0)


def class_weights(split: str) -> torch.Tensor:
    counts = np.bincount(fake_protocol(split).label.values, minlength=2).astype(float)
    weights = counts.sum() / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32)

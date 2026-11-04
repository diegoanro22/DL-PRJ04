"""Dataset y dataloaders para los protocolos CSV de src/data/README.md."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample

from src.data import augment

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600  # 4 s
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1
PROTOCOLS_DIR = Path(__file__).resolve().parents[2] / "protocols"


def protocol_path(split: str) -> Path:
    """Ruta del CSV de un split, p. ej. "asv19_train" -> protocols/asv19_train.csv."""
    return PROTOCOLS_DIR / f"{split}.csv"


def load_protocol(split: str) -> pd.DataFrame:
    path = protocol_path(split)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"utt_id", "dataset", "path", "label", "attack", "speaker", "language", "codec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path}: protocolo vacío")
    for col in ("utt_id", "dataset", "path"):
        if df[col].str.strip().eq("").any():
            raise ValueError(f"{path}: {col} contiene valores vacíos")
    if df.utt_id.duplicated().any():
        raise ValueError(f"{path}: utt_id duplicado")
    if not df.label.isin(["0", "1"]).all():
        raise ValueError(f"{path}: label debe ser 0 (genuina) o 1 (sintética)")
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in df.path):
        raise ValueError(f"{path}: las rutas deben ser relativas al dataset, sin '..'")
    df["label"] = df.label.astype(int)
    return df


class AudioDataset(Dataset):
    def __init__(self, cfg: dict, split: str, train: bool):
        self.protocol = load_protocol(split)
        self.train = train
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", SAMPLE_RATE))
        self.num_samples = int(cfg.get("audio", {}).get("num_samples", NUM_SAMPLES))
        if self.sample_rate <= 0 or self.num_samples <= 0:
            raise ValueError("sample_rate y num_samples deben ser positivos")
        self.roots = {key: Path(value) for key, value in cfg.get("dataset_roots", {}).items()}
        missing = set(self.protocol.dataset) - self.roots.keys()
        if missing:
            raise ValueError(f"Faltan dataset_roots para {sorted(missing)}")
        self.augmentations = []
        if train:
            for name in ("telephony", "rawboost"):
                if cfg.get("augment", {}).get(name, False):
                    self.augmentations.append(getattr(augment, name))

    def __len__(self) -> int:
        return len(self.protocol)

    def __getitem__(self, index: int) -> dict:
        row = self.protocol.iloc[index]
        path = self.roots[row.dataset] / row.path
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        if not wav.numel() or not torch.isfinite(wav).all():
            raise ValueError(f"{path}: audio vacío o con muestras no finitas")
        if sr != self.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
        for fn in self.augmentations:
            wav = torch.from_numpy(np.asarray(fn(wav.numpy(), self.sample_rate), dtype=np.float32))
        if wav.ndim != 1 or not wav.numel() or not torch.isfinite(wav).all():
            raise ValueError(f"{path}: audio inválido después del preprocesamiento")
        # Mantener la amplitud original; limitar únicamente valores fuera de rango.
        wav = wav.clamp(-1, 1)
        if len(wav) < self.num_samples:
            wav = wav.repeat((self.num_samples + len(wav) - 1) // len(wav))
            start = 0
        elif self.train:
            start = int(torch.randint(len(wav) - self.num_samples + 1, ()).item())
        else:
            start = 0
        wav = wav[start:start + self.num_samples].contiguous()
        return {"wav": wav, "label": int(row.label), "utt_id": row.utt_id}


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(cfg: dict, split: str, train: bool) -> DataLoader:
    """DataLoader de un split.

    Cada batch es un dict {"wav": [B, 64600] float32, "label": [B] long, "utt_id": list[str]}.
    Con train=True: recorte aleatorio, augmentation según cfg["augment"] y shuffle.
    Con train=False: primeras 64,600 muestras, sin augmentation ni shuffle.
    """
    generator = torch.Generator().manual_seed(cfg.get("seed", 42))
    return DataLoader(
        AudioDataset(cfg, split, train),
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=str(cfg.get("device", "cpu")).startswith("cuda"),
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def class_weights(split: str) -> torch.Tensor:
    """Pesos [genuina, sintética] inversos a la frecuencia de cada clase en el split."""
    counts = np.bincount(load_protocol(split).label.to_numpy(), minlength=2)
    if (counts == 0).any():
        raise ValueError(f"{split}: se requieren ambas clases para calcular los pesos")
    return torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)

"""Utilidades compartidas: semillas, precisión, dispositivo e info del entorno."""

import platform
import random
import subprocess

import numpy as np
import torch

from src.config import REPO_ROOT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("El config pide CUDA pero no hay GPU disponible (usar --device cpu).")
    return torch.device(name)


def amp_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """dtype para torch.autocast; None significa fp32 (sin autocast). En CPU siempre fp32."""
    if device.type != "cuda":
        return None
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("Esta GPU no soporta bf16: usar precision fp16 en el config de la máquina.")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "cpu"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "-"

"""Pruebas del modelo A (LFCC + LCNN) y del Grad-CAM."""

import sys

import pytest
import torch

from src.config import load_config
from src.data.dataset import NUM_SAMPLES
from src.models import build_model
from src.models.lfcc_cnn import LFCCCNN


def test_lfcc_cnn_forma_y_gradiente():
    model = build_model(load_config("configs/lfcc_cnn.yaml", "laptop"))
    assert isinstance(model, LFCCCNN)
    logits = model(torch.randn(2, NUM_SAMPLES))
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert model.head[-1].weight.grad is not None


def test_lfcc_cnn_parametros_razonables():
    model = build_model(load_config("configs/lfcc_cnn.yaml", "laptop"))
    n = sum(p.numel() for p in model.parameters())
    # LCNN ligera: del orden de 10^5–10^6, nunca cientos de M.
    assert 50_000 < n < 5_000_000


def test_train_py_entrena_lfcc(tmp_path, monkeypatch):
    from src import train

    original = train.load_config

    def patched(path, machine):
        cfg = original(path, machine)
        cfg["output_root"] = str(tmp_path)
        cfg["train"]["batch_size"] = 4
        return cfg

    monkeypatch.setattr(train, "load_config", patched)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--config", "configs/lfcc_cnn.yaml", "--fake-data", "--device", "cpu",
        "--epochs", "1", "--limit-train-batches", "2", "--limit-dev-batches", "2",
        "--name", "A_smoke", "--precision", "fp32",
    ])
    train.main()
    assert (tmp_path / "A_smoke" / "best.pt").exists()


def test_gradcam_lfcc_smoke():
    from src.explain.gradcam_lfcc import gradcam

    model = build_model(load_config("configs/lfcc_cnn.yaml", "laptop"))
    wav = torch.randn(1, NUM_SAMPLES)
    lfcc, cam = gradcam(model, wav)
    assert lfcc.ndim == 2
    assert cam.ndim == 2
    assert 0 <= cam.min() <= cam.max() <= 1

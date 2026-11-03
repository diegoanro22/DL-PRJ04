"""Exporta un modelo a ONNX para servirlo en CPU con ONNX Runtime (API y HF Spaces).

El modelo ONNX recibe la forma de onda [B, 64600] float32 y devuelve logits [B, 2]. Después
de exportar se verifica que ONNX Runtime dé los mismos logits que PyTorch.

Uso:
    uv run python -m src.export_onnx --source runs/C1/best.pt            # -> runs/C1/model.onnx
    uv run python -m src.export_onnx --source configs/aasist_l.yaml --out /tmp/b.onnx  # sin entrenar
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import load_config
from src.data.dataset import NUM_SAMPLES
from src.models import build_model


def load_model(source: str | Path, machine: str = "laptop") -> tuple[nn.Module, dict, str]:
    """Carga un checkpoint (.pt) o, para medir latencia antes de entrenar, un config (.yaml)
    con pesos aleatorios. Devuelve (modelo en eval y CPU, cfg, nombre)."""
    source = Path(source)
    if source.suffix in {".yaml", ".yml"}:
        cfg = load_config(source, machine)
        model = build_model(cfg)
        name = source.stem
    else:
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"])
        name = source.parent.name
    return model.eval(), cfg, name


def export(model: nn.Module, out: str | Path, atol: float = 1e-4) -> float:
    """Exporta con batch dinámico y devuelve la diferencia máxima de logits contra PyTorch."""
    import onnxruntime as ort

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = torch.randn(1, NUM_SAMPLES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (example,), str(out), input_names=["wav"], output_names=["logits"],
            dynamic_axes={"wav": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18, dynamo=False,
        )

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    check = torch.randn(3, NUM_SAMPLES)
    with torch.no_grad():
        expected = model(check).numpy()
    got = session.run(None, {"wav": check.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    if diff > atol:
        raise RuntimeError(f"ONNX no coincide con PyTorch: diferencia máxima {diff:.2e}")
    return diff


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta un modelo a ONNX y lo verifica.")
    p.add_argument("--source", required=True, help="checkpoint .pt o config .yaml")
    p.add_argument("--out", help="ruta del .onnx (por defecto, model.onnx junto al checkpoint)")
    args = p.parse_args()

    model, _, name = load_model(args.source)
    out = Path(args.out) if args.out else Path(args.source).with_name("model.onnx")
    diff = export(model, out)
    size_mb = out.stat().st_size / 1024**2
    print(f"{name}: {out} ({size_mb:.1f} MB), diferencia máxima con PyTorch {diff:.1e}")


if __name__ == "__main__":
    main()

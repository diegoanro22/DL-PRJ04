"""Grad-CAM sobre el mapa LFCC del modelo A (LFCC + LCNN).

Engancha la última conv del backbone, propaga el gradiente de la clase sintética
y produce un mapa de saliencia [frames, coeff] alineado con el LFCC.

Uso:
    uv run python -m src.explain.gradcam_lfcc \\
        --checkpoint runs/A1/best.pt --audio data/demo.wav --out results/figures/gradcam.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

from src.config import REPO_ROOT
from src.data.dataset import NUM_SAMPLES, SAMPLE_RATE
from src.export_onnx import load_model
from src.models.lfcc_cnn import LFCCCNN


def _prepare_wav(path: Path, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SAMPLE_RATE:
        wav = resample(wav, sr, SAMPLE_RATE)
    if wav.numel() < num_samples:
        reps = int(np.ceil(num_samples / max(wav.numel(), 1)))
        wav = wav.repeat(reps)
    return wav[:num_samples].unsqueeze(0)


def gradcam(model: LFCCCNN, wav: torch.Tensor, class_idx: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (lfcc_map [F,C], cam [F,C] normalizado 0–1)."""
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    # Última Conv2d del backbone (antes del dropout final).
    target_layer = next(
        m for m in reversed(list(model.backbone.modules())) if isinstance(m, torch.nn.Conv2d)
    )

    def fwd_hook(_module, _inp, out):
        activations["value"] = out

    def bwd_hook(_module, _gin, gout):
        gradients["value"] = gout[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        wav = wav.detach().requires_grad_(True)
        logits = model(wav)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        acts = activations["value"]  # [1, C, H, W]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=acts.shape[-2:], mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        with torch.no_grad():
            lfcc = model.lfcc_map(wav).squeeze().cpu().numpy()  # [F, C]
    finally:
        h1.remove()
        h2.remove()
    return lfcc, cam


def plot_gradcam(lfcc: np.ndarray, cam: np.ndarray, out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].imshow(lfcc.T, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_ylabel("coef LFCC")
    axes[0].set_title(title)
    axes[1].imshow(cam.T if cam.ndim == 2 else cam, aspect="auto", origin="lower", cmap="hot")
    axes[1].set_ylabel("saliencia")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM sobre LFCC del modelo A.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "figures" / "gradcam_lfcc.png")
    p.add_argument("--class-idx", type=int, default=1, help="1 = sintética")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, cfg, name = load_model(args.checkpoint)
    if not isinstance(model, LFCCCNN):
        raise SystemExit(f"Grad-CAM LFCC solo aplica al modelo A; recibido {type(model)}")
    wav = _prepare_wav(Path(args.audio), int(cfg.get("audio", {}).get("num_samples", NUM_SAMPLES)))
    lfcc, cam = gradcam(model, wav, class_idx=args.class_idx)
    # Redimensiona el CAM al tamaño del LFCC si el pooling lo achicó.
    if cam.shape != lfcc.shape:
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_t = torch.nn.functional.interpolate(
            cam_t, size=lfcc.shape, mode="bilinear", align_corners=False,
        )
        cam = cam_t.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    plot_gradcam(lfcc, cam, args.out, f"Grad-CAM LFCC — {name}")
    print(f"Guardado: {args.out}")


if __name__ == "__main__":
    main()

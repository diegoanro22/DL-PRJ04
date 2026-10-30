"""Prueba de memoria: ¿cabe el fine-tuning de XLS-R 300M en esta GPU?

Corre unos pasos de entrenamiento con audio aleatorio de 4 s y reporta el pico de VRAM
y el tiempo por paso. Mide el peor caso: fine-tuning completo con Adam y sin layerdrop.
La cabeza AASIST real pesa menos de 1 MB, así que aquí se usa una cabeza pequeña
equivalente; casi toda la memoria se va en XLS-R, sus gradientes y el estado de Adam.

Uso (desde la raíz del repo):
    uv run python scripts/check_gpu_memory.py                      # laptop: batch 4, bf16
    uv run python scripts/check_gpu_memory.py --batch-size 8 --precision fp16   # Kaggle
"""

import argparse
import time

import torch
from torch import nn
from transformers import Wav2Vec2Model

NUM_SAMPLES = 64600  # 4 s a 16 kHz, igual que en los configs
SSL_NAME = "facebook/wav2vec2-xls-r-300m"


class TinyHead(nn.Module):
    """Sustituto de la cabeza AASIST para medir memoria (mismo orden de parámetros)."""

    def __init__(self, dim_in: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim_in, 128), nn.SELU(), nn.Linear(128, 64), nn.SELU())
        self.out = nn.Linear(64, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(hidden).mean(dim=1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--no-checkpointing", action="store_true", help="desactiva gradient checkpointing")
    p.add_argument("--freeze-first-layers", type=int, default=0,
                   help="congela el extractor CNN y las primeras N capas transformer (plan B)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible: revisar el driver o la instalación de PyTorch.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / 1024**3
    print(f"GPU: {props.name} | {total_gb:.1f} GB | capability {props.major}.{props.minor}")
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("Esta GPU no soporta bf16: usar --precision fp16.")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]

    ssl = Wav2Vec2Model.from_pretrained(SSL_NAME, layerdrop=0.0, apply_spec_augment=False)
    if not args.no_checkpointing:
        ssl.gradient_checkpointing_enable()
    if args.freeze_first_layers > 0:
        ssl.freeze_feature_encoder()
        for layer in ssl.encoder.layers[: args.freeze_first_layers]:
            layer.requires_grad_(False)
    head = TinyHead(ssl.config.hidden_size)
    ssl.to(device).train()
    head.to(device).train()

    trainable = sum(p.numel() for p in ssl.parameters() if p.requires_grad)
    print(f"Parámetros entrenables de XLS-R: {trainable / 1e6:.0f} M")

    optimizer = torch.optim.Adam([
        {"params": [p for p in ssl.parameters() if p.requires_grad], "lr": 1e-6},
        {"params": head.parameters(), "lr": 1e-4},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    loss_fn = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    for step in range(args.steps):
        wav = torch.randn(args.batch_size, NUM_SAMPLES, device=device)
        labels = torch.randint(0, 2, (args.batch_size,), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = head(ssl(wav).last_hidden_state)
                loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            # Justo el caso que esta prueba debe diagnosticar: se reporta en vez de caerse.
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"\nResultado\n  batch {args.batch_size}, {args.precision}: se quedó sin memoria en el "
                  f"paso {step + 1} (pico antes del fallo: {peak_gb:.2f} GB de {total_gb:.1f} GB)")
            print("  veredicto: NO CABE. Probar un batch menor, --freeze-first-layers o quitar "
                  "--no-checkpointing.")
            raise SystemExit(1)
        step_times.append(time.perf_counter() - t0)
        print(f"  paso {step + 1}/{args.steps}: {step_times[-1]:.2f} s")

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    # El primer paso incluye la compilación de kernels; no cuenta para el promedio.
    steady = step_times[1:] or step_times
    sec_per_step = sum(steady) / len(steady)

    print("\nResultado")
    print(f"  batch {args.batch_size}, {args.precision}, checkpointing {'no' if args.no_checkpointing else 'sí'}")
    print(f"  pico asignado: {peak_gb:.2f} GB | pico reservado: {reserved_gb:.2f} GB de {total_gb:.1f} GB")
    print(f"  tiempo por paso: {sec_per_step:.2f} s")

    # ASVspoof 2019 LA train tiene 25,380 audios: estimación de una época.
    epoch_min = 25380 / args.batch_size * sec_per_step / 60
    print(f"  época estimada (25,380 audios): {epoch_min:.0f} min")
    margin = total_gb - reserved_gb
    verdict = "CABE" if margin > 0.5 else "AL LÍMITE" if margin > 0 else "NO CABE"
    print(f"  veredicto: {verdict} (margen {margin:.2f} GB)")


if __name__ == "__main__":
    main()

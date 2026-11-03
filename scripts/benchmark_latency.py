"""Latencia, tamaño y costo por predicción de cada modelo, medidos en la misma máquina.

Para cada modelo mide, con un audio de 4 s:
- ONNX Runtime en CPU con `--threads` hilos (simula la instancia de nube donde se serviría),
- PyTorch en CPU con los mismos hilos,
- PyTorch en GPU (fp32, batch 1) y el throughput en GPU (batch de hasta 32),
y calcula el costo de cómputo por llamada (c_comp del modelo de valor) como
latencia p50 en ONNX CPU × precio por hora de la instancia / 3600.

La latencia no depende de los pesos, así que se puede medir con un config (pesos aleatorios)
antes de entrenar y repetir con el checkpoint final.

Uso:
    uv run python scripts/benchmark_latency.py --models configs/lfcc_cnn.yaml \\
        configs/aasist_l.yaml configs/xlsr_aasist.yaml
    uv run python scripts/benchmark_latency.py --models runs/B2/best.pt runs/C2/best.pt

Guarda (o actualiza por modelo) results/latency.csv.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT
from src.data.dataset import NUM_SAMPLES
from src.export_onnx import export, load_model

OUT_CSV = REPO_ROOT / "results" / "latency.csv"
# AWS c6i.large (2 vCPU, 4 GB), on-demand en us-east-1. Ajustable con --cpu-price-usd-hour.
DEFAULT_CPU_PRICE = 0.085


def time_fn(fn, runs: int, warmup: int = 3) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return np.array(out) * 1000  # ms


def benchmark(source: str, threads: int, runs: int, cpu_price: float) -> dict:
    import onnxruntime as ort

    model, _, name = load_model(source)
    n_params = sum(p.numel() for p in model.parameters())
    wav = torch.randn(1, NUM_SAMPLES)
    row = {"modelo": name, "fuente": source, "parametros_M": round(n_params / 1e6, 3),
           "hilos_cpu": threads}

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export(model, onnx_path)
        row["onnx_MB"] = round(onnx_path.stat().st_size / 1024**2, 1)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        feed = {"wav": wav.numpy()}
        ms = time_fn(lambda: session.run(None, feed), runs)
    row["onnx_cpu_p50_ms"], row["onnx_cpu_p95_ms"] = np.percentile(ms, [50, 95]).round(1)

    torch.set_num_threads(threads)
    with torch.no_grad():
        ms = time_fn(lambda: model(wav), runs)
    row["torch_cpu_p50_ms"], row["torch_cpu_p95_ms"] = np.percentile(ms, [50, 95]).round(1)

    if torch.cuda.is_available():
        gpu_model = model.cuda()
        gpu_wav = wav.cuda()

        def run_gpu(x):
            gpu_model(x)
            torch.cuda.synchronize()

        with torch.no_grad():
            ms = time_fn(lambda: run_gpu(gpu_wav), runs, warmup=10)
            row["torch_gpu_p50_ms"], row["torch_gpu_p95_ms"] = np.percentile(ms, [50, 95]).round(1)
            # Throughput: el batch más grande (hasta 32) que quepa en la GPU.
            size = 32
            while size >= 1:
                try:
                    batch = torch.randn(size, NUM_SAMPLES, device="cuda")
                    ms_batch = time_fn(lambda b=batch: run_gpu(b), max(runs // 5, 5), warmup=3)
                    break
                except torch.OutOfMemoryError:
                    batch = None
                    torch.cuda.empty_cache()
                    size //= 2
        row["gpu_batch"] = size
        row["gpu_audios_por_s"] = round(size / (np.median(ms_batch) / 1000), 1)
        row["gpu"] = torch.cuda.get_device_name()
        model.cpu()
        torch.cuda.empty_cache()

    # c_comp: segundos de CPU por llamada × precio de la instancia por segundo.
    seconds = row["onnx_cpu_p50_ms"] / 1000
    row["usd_por_1000_llamadas"] = round(seconds * cpu_price / 3600 * 1000, 5)
    row["precio_cpu_usd_h"] = cpu_price
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="configs .yaml o checkpoints .pt")
    p.add_argument("--threads", type=int, default=2, help="hilos de CPU (vCPU de la instancia)")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--cpu-price-usd-hour", type=float, default=DEFAULT_CPU_PRICE)
    p.add_argument("--out", default=str(OUT_CSV))
    args = p.parse_args()

    rows = []
    for source in args.models:
        print(f"Midiendo {source}...")
        row = benchmark(source, args.threads, args.runs, args.cpu_price_usd_hour)
        rows.append(row)
        print(f"  ONNX CPU p50 {row['onnx_cpu_p50_ms']} ms, p95 {row['onnx_cpu_p95_ms']} ms | "
              f"USD {row['usd_por_1000_llamadas']} por 1,000 llamadas | {row['onnx_MB']} MB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.modelo.isin(new.modelo)], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()

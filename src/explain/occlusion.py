"""Explicabilidad por oclusión para cualquier modelo (pensado para XLS-R + AASIST).

Tapa partes del audio y mide cuánto baja la evidencia de "sintética" (log-odds del score):

- Frecuencia: anula una banda en el espectro (FFT) y promedia sobre audios de eval por ataque.
  Responde "en qué rango de frecuencias está la evidencia" y si sobrevive al canal telefónico
  (300-3400 Hz).
- Tiempo: silencia ventanas de 0.25 s de un audio y dibuja la caída sobre la forma de onda.

No necesita gradientes ni conocer la arquitectura, así que vale igual para C0 y C2.

Uso:
    uv run python -m src.explain.occlusion --checkpoint runs/C2/best.pt --per-attack 40
    uv run python -m src.explain.occlusion --checkpoint runs/C2/best.pt --audio clon.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import REPO_ROOT, apply_machine
from src.data.dataset import NUM_SAMPLES, SAMPLE_RATE, AudioDataset
from src.export_onnx import load_model

BANDS_HZ = [(0, 300), (300, 1000), (1000, 2000), (2000, 3400), (3400, 5000), (5000, 8000)]
WINDOW_S = 0.25


def log_odds(model: torch.nn.Module, wav: torch.Tensor, device: str) -> np.ndarray:
    """log P(sintética) - log P(genuina) por audio; no se satura como la probabilidad."""
    with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                         enabled=device.startswith("cuda")):
        logits = model(wav.to(device)).float()
    return (logits[:, 1] - logits[:, 0]).cpu().numpy()


def band_stop(wav: torch.Tensor, lo: float, hi: float, sr: int = SAMPLE_RATE) -> torch.Tensor:
    spec = torch.fft.rfft(wav, dim=-1)
    freqs = torch.fft.rfftfreq(wav.shape[-1], 1 / sr)
    spec[..., (freqs >= lo) & (freqs < hi)] = 0
    return torch.fft.irfft(spec, n=wav.shape[-1], dim=-1)


def time_mask(wav: torch.Tensor, start: int, length: int) -> torch.Tensor:
    out = wav.clone()
    out[..., start:start + length] = 0
    return out


def frequency_occlusion(model, cfg: dict, split: str, per_attack: int, device: str,
                        batch: int = 16, seed: int = 0) -> pd.DataFrame:
    """Caída media de log-odds al quitar cada banda, por ataque (y en genuinas)."""
    ds = AudioDataset(cfg, split, train=False)
    proto = ds.protocol
    rng = np.random.default_rng(seed)
    rows = []
    for attack, group in proto.groupby("attack"):
        idx = rng.choice(group.index.to_numpy(), size=min(per_attack, len(group)), replace=False)
        for i in range(0, len(idx), batch):
            wav = torch.stack([ds[int(j)]["wav"] for j in idx[i:i + batch]])
            base = log_odds(model, wav, device)
            for lo, hi in BANDS_HZ:
                drop = base - log_odds(model, band_stop(wav, lo, hi), device)
                for b, d in zip(base, drop):
                    rows.append({"ataque": attack, "banda": f"{lo}-{hi} Hz", "base": b, "caida": d})
        print(f"  {attack}: {len(idx)} audios", flush=True)
    return pd.DataFrame(rows)


def plot_frequency(df: pd.DataFrame, out: Path, title: str) -> None:
    bands = [f"{lo}-{hi} Hz" for lo, hi in BANDS_HZ]
    spoof = df[df.ataque != "-"].groupby("banda").caida.mean().reindex(bands)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2f5d50" if (lo >= 300 and hi <= 3400) else "#9aa79f" for lo, hi in BANDS_HZ]
    ax.bar(bands, spoof.values, color=colors)
    ax.set_ylabel("caída media de log-odds\n(audios sintéticos)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=20)
    ax.text(0.99, 0.97, "verde: banda telefónica (300-3400 Hz)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#2f5d50")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def temporal_occlusion(model, wav: torch.Tensor, device: str) -> tuple[np.ndarray, float]:
    win = int(WINDOW_S * SAMPLE_RATE)
    starts = list(range(0, wav.shape[-1] - win + 1, win // 2))
    base = float(log_odds(model, wav[None], device)[0])
    masked = torch.stack([time_mask(wav, s, win) for s in starts])
    drops = base - log_odds(model, masked, device)
    curve = np.zeros(wav.shape[-1])
    counts = np.zeros(wav.shape[-1])
    for s, d in zip(starts, drops):
        curve[s:s + win] += d
        counts[s:s + win] += 1
    return curve / np.maximum(counts, 1), base


def plot_temporal(wav: np.ndarray, curve: np.ndarray, base: float, out: Path, title: str) -> None:
    t = np.arange(len(wav)) / SAMPLE_RATE
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    a1.plot(t, wav, color="#1d2b3a", linewidth=0.4)
    a1.set_ylabel("amplitud")
    a1.set_title(f"{title} (log-odds sin tapar: {base:.1f})")
    a2.fill_between(t, curve, color="#2f5d50")
    a2.axhline(0, color="black", linewidth=0.8)
    a2.set_ylabel("caída de log-odds\nal tapar 0.25 s")
    a2.set_xlabel("segundos")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_audio(path: Path) -> torch.Tensor:
    import soundfile as sf
    from torchaudio.functional import resample
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SAMPLE_RATE:
        wav = resample(wav, sr, SAMPLE_RATE)
    if wav.numel() < NUM_SAMPLES:
        wav = wav.repeat(-(-NUM_SAMPLES // wav.numel()))
    return wav[:NUM_SAMPLES].clamp(-1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oclusión en frecuencia y tiempo.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--machine", default="laptop")
    p.add_argument("--split", default="asv19_eval")
    p.add_argument("--per-attack", type=int, default=40)
    p.add_argument("--audio", type=Path, help="un audio para la oclusión temporal")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, cfg, name = load_model(args.checkpoint, args.machine)
    cfg = apply_machine(cfg, args.machine)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if args.audio:
        wav = load_audio(args.audio)
        curve, base = temporal_occlusion(model, wav, device)
        out = args.out_dir / f"{name}_oclusion_tiempo_{args.audio.stem}.png"
        plot_temporal(wav.numpy(), curve, base, out, f"Oclusión temporal, {name}")
        print(f"Guardado: {out}")
        return
    df = frequency_occlusion(model, cfg, args.split, args.per_attack, device)
    csv = REPO_ROOT / "results" / "metrics" / f"{name}__oclusion_frecuencia.csv"
    df.to_csv(csv, index=False)
    table = df.pivot_table(index="ataque", columns="banda", values="caida", aggfunc="mean")
    print(table[[f"{lo}-{hi} Hz" for lo, hi in BANDS_HZ]].round(2).to_string())
    out = args.out_dir / f"{name}_oclusion_frecuencia.png"
    plot_frequency(df, out, f"Oclusión por banda de frecuencia, {name}")
    print(f"Guardado: {csv}\nGuardado: {out}")


if __name__ == "__main__":
    main()

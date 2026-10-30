"""Evaluación de un checkpoint sobre uno o más splits.

Genera el score P(sintética) de cada audio, calcula EER, AUC (con IC 95 % por bootstrap) y
min t-DCF cuando hay scores ASV (ASVspoof 2019 dev/eval), y desglosa el EER por ataque, códec
e idioma. Guarda:
    results/scores/<run>__<split>.csv    utt_id, score, label (lo usa el modelo de valor)
    results/metrics/<run>__<split>.json  métricas y desgloses

Uso:
    uv run python -m src.evaluate --checkpoint runs/<run>/best.pt --splits asv19_eval --machine laptop
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import REPO_ROOT, apply_machine
from src.data.dataset import LABEL_SPOOF, protocol_path
from src.metrics import (
    bootstrap_ci,
    compute_auc,
    compute_eer,
    compute_min_tdcf,
    eer_by_attack,
    eer_by_group,
    load_asv_scores,
)
from src.models import build_model
from src.utils import amp_dtype, get_device

RESULTS_DIR = REPO_ROOT / "results"
# Scores ASV oficiales de ASVspoof 2019 LA, relativos a dataset_roots["asv19"].
ASV_SCORE_FILES = {
    "asv19_dev": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
    "asv19_eval": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    limit_batches: int | None = None,
) -> pd.DataFrame:
    """Score P(sintética) por audio: DataFrame con utt_id, score, label."""
    model.eval()
    ids, scores, labels = [], [], []
    for i, batch in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        wav = batch["wav"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(wav)
        prob = torch.softmax(logits.float(), dim=1)[:, LABEL_SPOOF]
        ids.extend(batch["utt_id"])
        scores.append(prob.cpu().numpy())
        labels.append(batch["label"].numpy())
    return pd.DataFrame({
        "utt_id": ids,
        "score": np.concatenate(scores) if scores else np.array([]),
        "label": np.concatenate(labels) if labels else np.array([], dtype=int),
    })


def score_metrics(
    df: pd.DataFrame,
    protocol: pd.DataFrame | None = None,
    asv_file: Path | None = None,
    n_bootstrap: int = 1000,
) -> dict:
    """Métricas de un DataFrame de scores; con `protocol` agrega los desgloses."""
    s, y = df.score.values, df.label.values
    eer, threshold = compute_eer(s, y)
    out = {
        "n": len(df),
        "n_genuinas": int((y == 0).sum()),
        "n_sinteticas": int((y == 1).sum()),
        "eer": eer,
        "eer_ic95": bootstrap_ci(s, y, lambda a, b: compute_eer(a, b)[0], n=n_bootstrap),
        "eer_umbral": threshold,
        "auc": compute_auc(s, y),
        "auc_ic95": bootstrap_ci(s, y, compute_auc, n=n_bootstrap),
        "min_tdcf": None,
    }
    if asv_file is not None and asv_file.exists():
        keys, asv_scores = load_asv_scores(asv_file)
        out["min_tdcf"] = compute_min_tdcf(s, y, keys, asv_scores)

    if protocol is not None:
        meta = df.merge(protocol.drop(columns=["label"], errors="ignore"), on="utt_id", how="left")
        out["por_ataque"] = eer_by_attack(meta).to_dict("records") if "attack" in meta else []
        for col in ("codec", "language"):
            if col in meta and meta[col].nunique() > 1:
                out[f"por_{col}"] = eer_by_group(meta, col).to_dict("records")
    return out


def _fmt(x: float | None) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f} %"


def print_summary(split: str, m: dict) -> None:
    lo, hi = m["eer_ic95"]
    print(f"\n[{split}] {m['n']} audios ({m['n_genuinas']} genuinas, {m['n_sinteticas']} sintéticas)")
    print(f"  EER   {_fmt(m['eer'])}  (IC 95 %: {_fmt(lo)} a {_fmt(hi)})")
    print(f"  AUC   {m['auc']:.4f}")
    if m["min_tdcf"] is not None:
        print(f"  min t-DCF {m['min_tdcf']:.4f}")
    for row in m.get("por_ataque", []):
        print(f"    {row['grupo']:>6}: EER {_fmt(row['eer'])}  ({row['n_sinteticas']} sintéticas)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa un checkpoint sobre uno o más splits.")
    p.add_argument("--checkpoint", required=True, help="best.pt o last.pt de una corrida")
    p.add_argument("--splits", nargs="+", required=True, help="p. ej. asv19_eval asv21_eval")
    p.add_argument("--machine", default="laptop")
    p.add_argument("--device", help="sobrescribe el device de la máquina (p. ej. cpu)")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"],
                   help="sobrescribe la precisión de la máquina")
    p.add_argument("--name", help="nombre para los archivos de salida (por defecto, la corrida)")
    p.add_argument("--bootstrap", type=int, default=1000, help="remuestreos para los IC (0 = sin IC)")
    p.add_argument("--limit-batches", type=int, help="solo para pruebas rápidas")
    p.add_argument("--fake-data", action="store_true", help="dataset sintético para pruebas de humo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = apply_machine(ckpt["cfg"], args.machine)
    if args.device:
        cfg["device"] = args.device
    if args.precision:
        cfg["precision"] = args.precision
    device = get_device(cfg["device"])
    dtype = amp_dtype(cfg["precision"], device)
    run_name = args.name or ckpt_path.parent.name

    if args.fake_data:
        from src.data.fake import build_dataloader
        from src.data.fake import fake_protocol as load_protocol
    else:
        from src.data.dataset import build_dataloader

        def load_protocol(split: str) -> pd.DataFrame:
            return pd.read_csv(protocol_path(split))

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Las pruebas de humo se guardan junto al checkpoint, no en results/.
    out_dir = ckpt_path.parent / "eval" if args.fake_data else RESULTS_DIR
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        loader = build_dataloader(cfg, split, train=False)
        df = predict(model, loader, device, dtype, args.limit_batches)
        asv_file = None
        if split in ASV_SCORE_FILES and "asv19" in cfg.get("dataset_roots", {}):
            asv_file = Path(cfg["dataset_roots"]["asv19"]) / ASV_SCORE_FILES[split]
        metrics = score_metrics(df, load_protocol(split), asv_file, args.bootstrap)
        metrics.update({"run": run_name, "split": split, "checkpoint": str(ckpt_path),
                        "epoch": ckpt.get("epoch")})

        df.to_csv(out_dir / "scores" / f"{run_name}__{split}.csv", index=False)
        with open(out_dir / "metrics" / f"{run_name}__{split}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)
        print_summary(split, metrics)


if __name__ == "__main__":
    main()

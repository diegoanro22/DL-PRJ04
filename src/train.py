"""Entrenamiento genérico para los tres modelos.

Combina el config del modelo con el de la máquina, entrena con bf16/fp16 y acumulación de
gradientes, mide el EER en dev al final de cada época, guarda checkpoints (last.pt para
retomar y best.pt con el mejor EER), se detiene si el EER deja de mejorar y agrega una fila a
results/runs.csv.

Uso:
    uv run python -m src.train --config configs/aasist_l.yaml --machine laptop
    uv run python -m src.train --resume runs/<corrida>            # retomar (p. ej. en Kaggle)
    uv run python -m src.train --config configs/debug.yaml --fake-data --device cpu   # prueba
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.config import REPO_ROOT, load_config, save_config
from src.evaluate import predict
from src.metrics import compute_auc, compute_eer
from src.models import build_model
from src.utils import amp_dtype, device_name, get_device, git_commit, seed_everything

RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
RUNS_CSV_FIELDS = [
    "run", "fecha", "modelo", "config", "maquina", "gpu", "precision", "seed",
    "train_split", "dev_split", "batch_efectivo", "epocas", "mejor_epoca", "dev_eer", "dev_auc",
    "parametros_M", "horas_entrenamiento", "costo_usd", "commit", "notas",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena un modelo de detección de voz sintética.")
    p.add_argument("--config", help="config del modelo, p. ej. configs/aasist_l.yaml")
    p.add_argument("--machine", default="laptop", help="laptop o kaggle (configs/machines/)")
    p.add_argument("--name", help="nombre de la corrida (por defecto <modelo>_<fecha>)")
    p.add_argument("--resume", help="carpeta de una corrida para retomar desde last.pt")
    p.add_argument("--device", help="sobrescribe el device de la máquina (p. ej. cpu)")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"],
                   help="sobrescribe la precisión de la máquina")
    p.add_argument("--train-split", default="asv19_train")
    p.add_argument("--dev-split", default="asv19_dev")
    p.add_argument("--epochs", type=int, help="sobrescribe train.epochs")
    p.add_argument("--seed", type=int, help="sobrescribe seed")
    p.add_argument("--limit-train-batches", type=int, help="solo para pruebas rápidas")
    p.add_argument("--limit-dev-batches", type=int, help="solo para pruebas rápidas")
    p.add_argument("--notes", default="", help="nota libre que se guarda en runs.csv")
    p.add_argument("--fake-data", action="store_true", help="dataset sintético para pruebas de humo")
    args = p.parse_args()
    if not args.config and not args.resume:
        p.error("hace falta --config o --resume")
    return args


def build_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    if hasattr(model, "param_groups"):
        groups = model.param_groups(tcfg)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": tcfg["lr"]}]
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    name = tcfg.get("optimizer", "adam").lower()
    if name not in optimizers:
        raise ValueError(f"Optimizador desconocido: {name}")
    return optimizers[name](groups, weight_decay=tcfg.get("weight_decay", 0.0))


def build_scheduler(optimizer: torch.optim.Optimizer, tcfg: dict, total_steps: int):
    if tcfg.get("scheduler") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def build_loss(tcfg: dict, weights: torch.Tensor, device: torch.device) -> nn.Module:
    if tcfg.get("loss") == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights.to(device))
    return nn.CrossEntropyLoss()


def atomic_save(obj: dict, path: Path) -> None:
    """Guarda en un archivo temporal y renombra: si se corta la sesión no queda un .pt roto."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_run_row(row: dict) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    # --- Config y carpeta de la corrida --------------------------------------------------------
    resume_state = None
    if args.resume:
        run_dir = Path(args.resume)
        resume_state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        cfg = resume_state["cfg"]
        if args.device:
            cfg["device"] = args.device
        print(f"Retomando {run_dir} en la época {resume_state['epoch'] + 2}")
    else:
        cfg = load_config(args.config, args.machine)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        if args.device:
            cfg["device"] = args.device
        if args.precision:
            cfg["precision"] = args.precision
        cfg["train_split"], cfg["dev_split"] = args.train_split, args.dev_split
        name = args.name or f"{cfg['model']}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = Path(cfg["output_root"]) / name
        if (run_dir / "last.pt").exists():
            raise SystemExit(f"{run_dir} ya tiene una corrida: usar --resume {run_dir} u otro --name.")
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")

    tcfg = cfg["train"]
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    dtype = amp_dtype(cfg["precision"], device)

    if args.fake_data:
        from src.data.fake import build_dataloader, class_weights
    else:
        from src.data.dataset import build_dataloader, class_weights

    train_loader = build_dataloader(cfg, cfg["train_split"], train=True)
    dev_loader = build_dataloader(cfg, cfg["dev_split"], train=False)

    # --- Modelo, optimizador, scheduler --------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    accum = tcfg.get("grad_accum", 1)
    n_batches = len(train_loader)
    if args.limit_train_batches:
        n_batches = min(n_batches, args.limit_train_batches)
    steps_per_epoch = math.ceil(n_batches / accum)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg, steps_per_epoch * tcfg["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    loss_fn = build_loss(tcfg, class_weights(cfg["train_split"]), device)

    start_epoch, best_eer, best_epoch, bad_epochs, elapsed = 0, float("inf"), -1, 0, 0.0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch = resume_state["epoch"] + 1
        best_eer, best_epoch = resume_state["best_eer"], resume_state["best_epoch"]
        bad_epochs, elapsed = resume_state["bad_epochs"], resume_state["elapsed_sec"]

    gpu = device_name(device)
    print(f"Modelo {cfg['model']} ({n_params:.2f} M parámetros) en {gpu}, {cfg['precision']}")
    print(f"Batch {tcfg['batch_size']} x acumulación {accum} = {tcfg['batch_size'] * accum} efectivo; "
          f"{n_batches} batches por época")

    writer = SummaryWriter(run_dir / "tb")
    patience = tcfg.get("patience", tcfg["epochs"])
    epoch = start_epoch - 1

    # --- Ciclo de entrenamiento ----------------------------------------------------------------
    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, seen = 0.0, 0
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            wav = batch["wav"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(wav)
            loss = loss_fn(logits.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss no finita en la época {epoch + 1}, batch {i + 1}.")
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)

        dev = predict(model, dev_loader, device, dtype, args.limit_dev_batches)
        dev_eer, _ = compute_eer(dev.score.values, dev.label.values)
        dev_auc = compute_auc(dev.score.values, dev.label.values)
        epoch_sec = time.perf_counter() - t0
        elapsed += epoch_sec
        train_loss = running_loss / max(seen, 1)

        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("eer/dev", dev_eer, epoch + 1)
        writer.add_scalar("auc/dev", dev_auc, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch + 1)

        improved = dev_eer < best_eer
        if improved:
            best_eer, best_epoch, bad_epochs = dev_eer, epoch, 0
            atomic_save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                         "dev_eer": dev_eer}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "cfg": cfg,
            "epoch": epoch, "best_eer": best_eer, "best_epoch": best_epoch,
            "bad_epochs": bad_epochs, "elapsed_sec": elapsed,
        }, run_dir / "last.pt")

        mark = "  *mejor*" if improved else ""
        print(f"Época {epoch + 1}/{tcfg['epochs']}: loss {train_loss:.4f} | dev EER "
              f"{100 * dev_eer:.2f} % | AUC {dev_auc:.4f} | {epoch_sec / 60:.1f} min{mark}")
        if bad_epochs >= patience:
            print(f"Early stopping: {patience} épocas sin mejorar el EER en dev.")
            break
    writer.close()

    if best_epoch < 0:
        raise SystemExit("No se completó ninguna época.")

    # --- Registro de la corrida ----------------------------------------------------------------
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    dev = predict(model, dev_loader, device, dtype, args.limit_dev_batches)
    hours = elapsed / 3600
    row = {
        "run": run_dir.name,
        "fecha": f"{datetime.now():%Y-%m-%d %H:%M}",
        "modelo": cfg["model"],
        "config": cfg.get("config_path", "-"),
        "maquina": cfg["machine"],
        "gpu": gpu,
        "precision": cfg["precision"],
        "seed": cfg["seed"],
        "train_split": cfg["train_split"],
        "dev_split": cfg["dev_split"],
        "batch_efectivo": tcfg["batch_size"] * accum,
        "epocas": epoch + 1,
        "mejor_epoca": best_epoch + 1,
        "dev_eer": f"{compute_eer(dev.score.values, dev.label.values)[0]:.5f}",
        "dev_auc": f"{compute_auc(dev.score.values, dev.label.values):.5f}",
        "parametros_M": f"{n_params:.3f}",
        "horas_entrenamiento": f"{hours:.3f}",
        "costo_usd": f"{hours * cfg.get('gpu_price_usd_per_hour', 0):.2f}",
        "commit": git_commit(),
        "notas": args.notes,
    }
    print(f"\nListo. Mejor EER en dev: {100 * best_eer:.2f} % (época {best_epoch + 1}). "
          f"Tiempo: {hours * 60:.1f} min")
    if args.fake_data:
        # Las pruebas de humo no cuentan como resultados del proyecto.
        print(f"Checkpoints en {run_dir} (--fake-data: no se registra en runs.csv).")
    else:
        append_run_row(row)
        print(f"Checkpoints en {run_dir}. Fila agregada a {RUNS_CSV.relative_to(REPO_ROOT)}.")
    print(f"Evaluar: uv run python -m src.evaluate --checkpoint {run_dir / 'best.pt'} "
          f"--splits asv19_eval --machine {cfg['machine']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Genera un notebook listo para `kaggle kernels push` (A1/A2/B1/B2).

Uso:
  python scripts/make_kaggle_kernel.py --run B1 --out /tmp/kaggle-b1
  python scripts/make_kaggle_kernel.py --run A2 --out /tmp/kaggle-a2  # warm-start desde A1
  kaggle kernels push -p /tmp/kaggle-a2 --accelerator NvidiaTeslaT4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = {
    "A1": ("configs/lfcc_cnn.yaml", "A1", None),
    "A2": ("configs/lfcc_cnn_aug.yaml", "A2", "cisco890/dl-prj04-a1-lfcc-cnn"),
    "B1": ("configs/aasist_l.yaml", "B1", None),
    "B2": ("configs/aasist_l_aug.yaml", "B2", "cisco890/dl-prj04-b1"),
}

REPO_URL = "https://github.com/diegoanro22/DL-PRJ04.git"


def notebook(config: str, run_name: str, init_kernel: str | None) -> dict:
    setup = f'''import os, sys, subprocess, shutil
from pathlib import Path

REPO_URL = {REPO_URL!r}
BRANCH = "main"
CONFIG = {config!r}
RUN_NAME = {run_name!r}
MACHINE = "kaggle"
EVAL_SPLITS = ["asv19_dev", "asv19_eval"]
INIT_KERNEL = {init_kernel!r}  # kernel Kaggle cuyo output trae best.pt (warm-start)

WORK = Path("/kaggle/working")
REPO = WORK / "DL-PRJ04"
INP = Path("/kaggle/input")
print("input:", [p.name for p in INP.iterdir()] if INP.exists() else None, flush=True)

def find_asv19() -> Path:
    cands = [
        INP / "datasets" / "awsaf49" / "asvpoof-2019-dataset" / "LA" / "LA",
        INP / "asvpoof-2019-dataset" / "LA" / "LA",
    ]
    ds = INP / "datasets"
    if ds.is_dir():
        for owner in ds.iterdir():
            if not owner.is_dir():
                continue
            for slug in owner.iterdir():
                if not slug.is_dir():
                    continue
                name = slug.name.lower()
                if "asvpoof" in name or "asvspoof" in name:
                    cands += [slug / "LA" / "LA", slug / "LA"]
    for p in cands:
        if (p / "ASVspoof2019_LA_cm_protocols").is_dir() and (p / "ASVspoof2019_LA_train" / "flac").is_dir():
            return p
    raise FileNotFoundError("ASVspoof 2019 LA no encontrado. Candidatos: " + ", ".join(map(str, cands)))

def find_init_ckpt() -> Path | None:
    if not INIT_KERNEL:
        return None
    slug = INIT_KERNEL.split("/")[-1]
    roots = [INP / slug, INP / slug.replace("-", "_")]
    # Sin rglob: el FS de /kaggle/input cuelga con walks profundos.
    names = (
        "best.pt",
        "A1_artifacts/best.pt", "B1_artifacts/best.pt",
        "A2_artifacts/best.pt", "B2_artifacts/best.pt",
        "runs/A1/best.pt", "runs/B1/best.pt",
        "runs/A2/best.pt", "runs/B2/best.pt",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for name in names:
            p = root / name
            if p.is_file():
                return p
    print("WARN: no se encontró best.pt de", INIT_KERNEL, "- train from scratch", flush=True)
    return None

ASV19 = find_asv19()
print("ASV19:", ASV19, flush=True)
INIT_CKPT = find_init_ckpt()
print("init_from:", INIT_CKPT, flush=True)

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)])
os.chdir(REPO)
sys.path.insert(0, str(REPO))
print("commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(), flush=True)

machine = Path("configs/machines/kaggle.yaml")
lines = []
for line in machine.read_text().splitlines(True):
    if line.strip().startswith("asv19:"):
        lines.append(f"  asv19: {{ASV19}}\\n")
    elif line.strip().startswith("num_workers:"):
        # Train con 2 workers acelera aug; evaluate.py fuerza 0.
        lines.append("num_workers: 2\\n")
    else:
        lines.append(line)
machine.write_text("".join(lines))
print("kaggle.yaml asv19 + num_workers=2 OK", flush=True)

import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
'''

    deps = '''import subprocess
print("installing deps...", flush=True)
subprocess.check_call([
    "pip", "install", "-q",
    "pandas", "scikit-learn", "soundfile", "librosa", "pyyaml",
    "tqdm", "tensorboard", "matplotlib",
])
print("deps ok", flush=True)
'''

    protocols = '''import subprocess
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/machines/kaggle.yaml").read_text())
root = str(Path(cfg["dataset_roots"]["asv19"]))
print("build_protocols --skip-file-check", root, flush=True)
subprocess.check_call([
    "python", "scripts/build_protocols.py", "--root", root, "--skip-file-check",
])
subprocess.check_call(["wc", "-l"] + sorted(map(str, Path("protocols").glob("asv19_*.csv"))))
'''

    train = '''import subprocess
from pathlib import Path

cmd = ["python", "-m", "src.train", "--config", CONFIG, "--machine", MACHINE, "--name", RUN_NAME]
if INIT_CKPT is not None:
    cmd += ["--init-from", str(INIT_CKPT)]
print(" ".join(cmd), flush=True)
subprocess.check_call(cmd)
assert Path(f"/kaggle/working/runs/{RUN_NAME}/best.pt").is_file()
'''

    evaluate = '''import subprocess, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
ckpt = WORK / "runs" / RUN_NAME / "best.pt"
assert ckpt.is_file(), ckpt

cmd = [
    "python", "-m", "src.evaluate",
    "--checkpoint", str(ckpt),
    "--splits", *EVAL_SPLITS,
    "--machine", MACHINE,
    "--name", RUN_NAME,
]
print(" ".join(cmd), flush=True)
subprocess.check_call(cmd)

out = WORK / f"{RUN_NAME}_artifacts"
if out.exists():
    shutil.rmtree(out)
out.mkdir()
shutil.copy2(ckpt, out / "best.pt")
for src in (Path("results/scores"), Path("results/metrics"), Path("results/runs.csv")):
    dest = out / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    elif src.is_file():
        shutil.copy2(src, dest)
print("Listo", out, flush=True)
subprocess.check_call(["find", str(out), "-type", "f"])
'''

    def cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    warm = f" Warm-start desde `{init_kernel}`." if init_kernel else ""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# DL-PRJ04 {run_name} train+eval\\n",
                    "\\n",
                    f"Config `{config}`. T4 + skip-file-check + eval_every/aug p.{warm}\\n",
                ],
            },
            cell(setup),
            cell(deps),
            cell(protocols),
            cell(train),
            cell(evaluate),
        ],
    }


def metadata(slug: str, title: str, init_kernel: str | None) -> dict:
    return {
        "id": f"cisco890/{slug}",
        "title": title,
        "code_file": f"{slug.replace('-', '_')}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["awsaf49/asvpoof-2019-dataset"],
        "competition_sources": [],
        "kernel_sources": [init_kernel] if init_kernel else [],
        "model_sources": [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", choices=sorted(RUNS), required=True)
    p.add_argument("--out", type=Path, required=True, help="Carpeta para kernels push")
    p.add_argument("--slug", default=None, help="Slug del kernel (default dl-prj04-<run>)")
    p.add_argument("--no-init", action="store_true", help="No usar warm-start ni kernel_sources")
    args = p.parse_args()

    config, run_name, init_kernel = RUNS[args.run]
    if args.no_init:
        init_kernel = None
    slug = args.slug or f"dl-prj04-{run_name.lower()}"
    nb_name = f"{slug.replace('-', '_')}.ipynb"
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    (out / nb_name).write_text(json.dumps(notebook(config, run_name, init_kernel), indent=1) + "\n")
    meta = metadata(slug, f"DL-PRJ04 {run_name}", init_kernel)
    meta["code_file"] = nb_name
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Wrote {out / nb_name}")
    print(f"Wrote {out / 'kernel-metadata.json'}")
    if init_kernel:
        print(f"Warm-start kernel_sources: {init_kernel}")
    print(f"Push: kaggle kernels push -p {out} --accelerator NvidiaTeslaT4")


if __name__ == "__main__":
    main()

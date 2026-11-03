"""Pruebas de humo de train.py y evaluate.py con --fake-data y el modelo debug."""

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src import evaluate, train

CUDA = torch.cuda.is_available()


def run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


def read_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def config(tmp_path):
    """Copia de configs/debug.yaml para las pruebas."""
    path = tmp_path / "debug.yaml"
    path.write_text(yaml.safe_dump(read_yaml("configs/debug.yaml")))
    return path


@pytest.fixture(autouse=True)
def runs_in_tmp(tmp_path, monkeypatch):
    """Manda las corridas y runs.csv a tmp para no ensuciar runs/ ni results/."""
    original = train.load_config

    def load_config(path, machine):
        cfg = original(path, machine)
        cfg["output_root"] = str(tmp_path / "runs")
        return cfg

    monkeypatch.setattr(train, "load_config", load_config)
    monkeypatch.setattr(train, "RUNS_CSV", tmp_path / "runs.csv")


def test_entrena_en_cpu(config, tmp_path, monkeypatch):
    run(train, monkeypatch, "--config", str(config), "--fake-data", "--device", "cpu",
        "--epochs", "2", "--name", "cpu")
    run_dir = tmp_path / "runs" / "cpu"
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()
    assert (run_dir / "config.yaml").exists()
    assert not (tmp_path / "runs.csv").exists()  # --fake-data no registra en runs.csv
    last = torch.load(run_dir / "last.pt", weights_only=False)
    assert last["epoch"] == 1
    assert last["best_eer"] < 0.1  # el tono de 4 kHz se aprende rápido


@pytest.mark.skipif(not CUDA, reason="requiere GPU")
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_entrena_en_gpu(config, tmp_path, monkeypatch, precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU sin bf16")
    run(train, monkeypatch, "--config", str(config), "--fake-data", "--precision", precision,
        "--epochs", "2", "--name", precision)
    cfg = read_yaml(tmp_path / "runs" / precision / "config.yaml")
    assert cfg["precision"] == precision
    assert torch.load(tmp_path / "runs" / precision / "last.pt", weights_only=False)["best_eer"] < 0.1


def test_retoma_desde_last(config, tmp_path, monkeypatch):
    run(train, monkeypatch, "--config", str(config), "--fake-data", "--device", "cpu",
        "--epochs", "3", "--name", "resume")
    last_path = tmp_path / "runs" / "resume" / "last.pt"
    # Simula que la sesión se cortó después de la primera época.
    state = torch.load(last_path, weights_only=False)
    state["epoch"] = 0
    torch.save(state, last_path)

    run(train, monkeypatch, "--resume", str(tmp_path / "runs" / "resume"), "--fake-data",
        "--device", "cpu")
    assert torch.load(last_path, weights_only=False)["epoch"] == 2


def test_no_pisa_una_corrida_existente(config, monkeypatch):
    run(train, monkeypatch, "--config", str(config), "--fake-data", "--device", "cpu",
        "--epochs", "1", "--name", "dup")
    with pytest.raises(SystemExit):
        run(train, monkeypatch, "--config", str(config), "--fake-data", "--device", "cpu",
            "--epochs", "1", "--name", "dup")


def test_evalua_y_guarda_scores(config, tmp_path, monkeypatch):
    run(train, monkeypatch, "--config", str(config), "--fake-data", "--device", "cpu",
        "--epochs", "1", "--name", "ev")
    ckpt = tmp_path / "runs" / "ev" / "best.pt"
    run(evaluate, monkeypatch, "--checkpoint", str(ckpt), "--splits", "asv19_eval",
        "--fake-data", "--device", "cpu", "--bootstrap", "50")

    out = tmp_path / "runs" / "ev" / "eval"
    scores = (out / "scores" / "ev__asv19_eval.csv").read_text().splitlines()
    assert scores[0] == "utt_id,score,label"
    assert len(scores) == 1 + 256
    metrics = json.loads((out / "metrics" / "ev__asv19_eval.json").read_text())
    assert metrics["n"] == 256
    assert metrics["eer"] < 0.1
    assert {row["grupo"] for row in metrics["por_ataque"]} <= {"A01", "A02", "A03", "A04"}
    assert "por_codec" in metrics


def test_migra_runs_csv_con_columnas_viejas(tmp_path, monkeypatch):
    # runs.csv hecho con una versión anterior: sin entrenables_M.
    old_fields = [f for f in train.RUNS_CSV_FIELDS if f != "entrenables_M"]
    path = tmp_path / "runs.csv"
    path.write_text(",".join(old_fields) + "\n" + ",".join(f"viejo_{f}" for f in old_fields) + "\n")

    train.append_run_row({f: f"nuevo_{f}" for f in train.RUNS_CSV_FIELDS})

    import pandas as pd
    df = pd.read_csv(path)
    assert list(df.columns) == train.RUNS_CSV_FIELDS
    assert df.loc[0, "horas_entrenamiento"] == "viejo_horas_entrenamiento"
    assert df.loc[0, "entrenables_M"] == "-"
    assert df.loc[1, "horas_entrenamiento"] == "nuevo_horas_entrenamiento"
    assert df.loc[1, "entrenables_M"] == "nuevo_entrenables_M"

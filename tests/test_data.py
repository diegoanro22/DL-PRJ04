"""Datos de prueba en FLAC: protocolo oficial → CSV → audio → entrenamiento/evaluación."""

import csv
import json
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from scripts import build_protocols
from src.data import dataset


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "LA"
    official = root / "ASVspoof2019_LA_cm_protocols"
    official.mkdir(parents=True)
    out = tmp_path / "protocols"
    monkeypatch.setattr(dataset, "PROTOCOLS_DIR", out)
    for split, suffix in build_protocols.SPLITS.items():
        audio = root / f"ASVspoof2019_LA_{split}" / "flac"
        audio.mkdir(parents=True)
        lines = []
        for i in range(4):
            utt = f"LA_{split}_{i}"
            label = "spoof" if i else "bonafide"
            attack = "A01" if i else "-"
            lines.append(f"LA_0001 {utt} - {attack} {label}\n")
            # Señal corta y sin normalización; las dos clases no dependen de datos externos.
            wav = np.linspace(-0.2, 0.2, 3000, dtype=np.float32)
            sf.write(audio / f"{utt}.flac", wav, 16000, subtype="PCM_24")
        (official / f"ASVspoof2019.LA.cm.{split}.{suffix}.txt").write_text("".join(lines))
    monkeypatch.setattr(sys, "argv", [
        "build_protocols", "--root", str(root), "--output-dir", str(out)])
    build_protocols.main()
    cfg = {
        "dataset_roots": {"asv19": str(root)}, "seed": 42, "num_workers": 0,
        "device": "cpu", "audio": {"sample_rate": 16000, "num_samples": 64600},
        "train": {"batch_size": 2}, "augment": {"telephony": False, "rawboost": False},
    }
    return root, out, cfg


def test_protocolos_rutas_etiquetas_y_pesos(corpus):
    root, out, _ = corpus
    for split in ("train", "dev", "eval"):
        df = pd.read_csv(out / f"asv19_{split}.csv", keep_default_na=False)
        assert list(df.columns) == build_protocols.FIELDS
        assert df.label.tolist() == [0, 1, 1, 1]
        assert df.attack.tolist() == ["-", "A01", "A01", "A01"]
        assert df.codec.tolist() == ["none"] * 4
        assert all((root / p).is_file() for p in df.path)
    torch.testing.assert_close(dataset.class_weights("asv19_train"), torch.tensor([2., 2/3]))


def test_repeticion_sin_normalizacion_y_contrato_del_batch(corpus):
    root, _, cfg = corpus
    loader = dataset.build_dataloader(cfg, "asv19_dev", train=False)
    assert isinstance(loader.sampler, SequentialSampler)
    batch = next(iter(loader))
    assert batch["wav"].shape == (2, 64600)
    assert batch["wav"].dtype == torch.float32
    assert batch["label"].dtype == torch.long
    assert batch["label"].tolist() == [0, 1]
    assert batch["utt_id"] == ["LA_dev_0", "LA_dev_1"]
    original, _ = sf.read(root / "ASVspoof2019_LA_dev/flac/LA_dev_0.flac", dtype="float32")
    expected = np.tile(original, 22)[:64600]
    np.testing.assert_array_equal(batch["wav"][0].numpy(), expected)
    torch.testing.assert_close(batch["wav"], next(iter(loader))["wav"])


def test_mono_remuestreo_y_recorte_fijo(corpus):
    root, _, cfg = corpus
    path = root / "ASVspoof2019_LA_dev/flac/LA_dev_0.flac"
    # Stereo 8 kHz: la media es 0.2. Audio largo para cubrir el recorte.
    stereo = np.tile(np.array([0.1, 0.3], dtype=np.float32), (40000, 1))
    sf.write(path, stereo, 8000, subtype="PCM_24")
    wav = dataset.AudioDataset(cfg, "asv19_dev", train=False)[0]["wav"]
    assert wav.shape == (64600,)
    np.testing.assert_allclose(wav[100:-100].numpy(), 0.2, atol=0.001)


def test_recorte_aleatorio_incluye_la_ultima_posicion(corpus, monkeypatch):
    root, _, cfg = corpus
    original = np.linspace(-0.5, 0.5, 65000, dtype=np.float32)
    path = root / "ASVspoof2019_LA_train/flac/LA_train_0.flac"
    sf.write(path, original, 16000, subtype="PCM_24")
    monkeypatch.setattr(torch, "randint", lambda high, size: torch.tensor(high - 1))
    wav = dataset.AudioDataset(cfg, "asv19_train", train=True)[0]["wav"]
    decoded, _ = sf.read(path, dtype="float32")
    np.testing.assert_array_equal(wav.numpy(), decoded[-64600:])


def test_shuffle_reproducible_con_workers(corpus):
    root, _, cfg = corpus
    wav = np.linspace(-0.5, 0.5, 70000, dtype=np.float32)
    sf.write(root / "ASVspoof2019_LA_train/flac/LA_train_0.flac", wav, 16000)
    cfg["num_workers"] = 2
    a = dataset.build_dataloader(cfg, "asv19_train", train=True)
    b = dataset.build_dataloader(cfg, "asv19_train", train=True)
    assert isinstance(a.sampler, RandomSampler)
    batches_a, batches_b = list(a), list(b)
    assert [x["utt_id"] for x in batches_a] == [x["utt_id"] for x in batches_b]
    for x, y in zip(batches_a, batches_b):
        torch.testing.assert_close(x["wav"], y["wav"])


def test_augmentation_solo_en_train(corpus, monkeypatch):
    _, _, cfg = corpus
    cfg["augment"] = {"telephony": True, "rawboost": True}
    calls = []

    def augmentation(wav, sr):
        calls.append(sr)
        return wav

    monkeypatch.setattr(dataset.augment, "telephony", augmentation)
    monkeypatch.setattr(dataset.augment, "rawboost", augmentation)
    dataset.AudioDataset(cfg, "asv19_dev", train=False)[0]
    assert calls == []
    dataset.AudioDataset(cfg, "asv19_train", train=True)[0]
    assert calls == [16000, 16000]


@pytest.mark.parametrize("change, message", [
    (lambda df: df.assign(label=7), "label"),
    (lambda df: df.assign(utt_id="duplicado"), "duplicado"),
    (lambda df: df.assign(path="../afuera.flac"), "relativas"),
    (lambda df: df.drop(columns="codec"), "columnas"),
    (lambda df: df.iloc[:0], "vacío"),
])
def test_rechaza_protocolos_invalidos(corpus, change, message):
    _, out, cfg = corpus
    path = out / "asv19_train.csv"
    change(pd.read_csv(path, keep_default_na=False)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        dataset.build_dataloader(cfg, "asv19_train", train=True)


def test_rechaza_una_sola_clase_y_raiz_faltante(corpus):
    _, out, cfg = corpus
    df = pd.read_csv(out / "asv19_train.csv", keep_default_na=False)
    df.assign(label=1).to_csv(out / "asv19_train.csv", index=False)
    with pytest.raises(ValueError, match="ambas clases"):
        dataset.class_weights("asv19_train")
    cfg["dataset_roots"] = {}
    with pytest.raises(ValueError, match="dataset_roots"):
        dataset.build_dataloader(cfg, "asv19_train", train=True)


def test_rechaza_audio_vacio(corpus):
    root, out, cfg = corpus
    # Un WAV sin cuadros tiene cabecera válida; un FLAC vacío no es decodificable.
    sf.write(root / "empty.wav", np.empty(0), 16000)
    path = out / "asv19_train.csv"
    df = pd.read_csv(path, keep_default_na=False)
    df.loc[0, "path"] = "empty.wav"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="vacío"):
        dataset.AudioDataset(cfg, "asv19_train", train=True)[0]


def test_generador_no_oculta_audio_faltante(corpus):
    root, _, _ = corpus
    (root / "ASVspoof2019_LA_train/flac/LA_train_0.flac").unlink()
    with pytest.raises(FileNotFoundError, match="LA_train_0.flac"):
        build_protocols.build_asv19_protocol(root, "train")


def test_generador_acepta_carpetas_cortas(corpus):
    root, _, _ = corpus
    (root / "ASVspoof2019_LA_train").rename(root / "train")
    rows = build_protocols.build_asv19_protocol(root, "train")
    assert rows[0]["path"] == "train/flac/LA_train_0.flac"


@pytest.mark.parametrize("line", ["LA_1 utt - A01 desconocida", "LA_1 utt", "LA_1 ../utt - A01 spoof"])
def test_generador_rechaza_formato_invalido(corpus, line):
    root, _, _ = corpus
    path = root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    path.write_text(line + "\n")
    with pytest.raises(ValueError):
        build_protocols.build_asv19_protocol(root, "train")


def test_entrenamiento_y_evaluacion_sin_fake_data(corpus, tmp_path, monkeypatch):
    from src import evaluate, train
    from src.config import load_yaml

    _, _, cfg = corpus
    debug = load_yaml("configs/debug.yaml")
    cfg = {**debug, **cfg, "train": {**debug["train"], "batch_size": 2, "epochs": 1},
           "machine": "prueba", "precision": "fp32", "output_root": str(tmp_path / "runs"),
           "gpu_price_usd_per_hour": 0.0}
    cfg["audio"]["num_samples"] = 2048
    monkeypatch.setattr(train, "load_config", lambda *_: cfg)
    monkeypatch.setattr(train, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(train, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train", "--config", "configs/debug.yaml", "--name", "audio_real"])
    train.main()
    checkpoint = tmp_path / "runs/audio_real/best.pt"
    assert checkpoint.is_file()
    with (tmp_path / "runs.csv").open() as f:
        assert next(csv.DictReader(f))["run"] == "audio_real"

    monkeypatch.setattr(evaluate, "apply_machine", lambda cfg, machine: cfg)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--checkpoint", str(checkpoint), "--splits", "asv19_dev", "--bootstrap", "0"])
    evaluate.main()
    scores = pd.read_csv(tmp_path / "results/scores/audio_real__asv19_dev.csv")
    assert scores.utt_id.tolist() == [f"LA_dev_{i}" for i in range(4)]
    assert scores.label.tolist() == [0, 1, 1, 1]
    assert scores.score.between(0, 1).all()
    metrics = json.loads((tmp_path / "results/metrics/audio_real__asv19_dev.json").read_text())
    assert metrics["n"] == 4

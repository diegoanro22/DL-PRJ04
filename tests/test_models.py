"""Pruebas de AASIST-L y XLS-R + AASIST (con un wav2vec2 chico para no bajar XLS-R)."""

import sys

import pytest
import torch
import yaml

from src import train
from src.config import load_config, load_yaml
from src.data.dataset import NUM_SAMPLES
from src.models import build_model
from src.models.xlsr_aasist import XLSRAASIST

# wav2vec2 de juguete: misma forma de salida que XLS-R (201 cuadros en 4 s), mucho más chico.
TINY_SSL = {
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
    "intermediate_size": 64, "conv_dim": [32] * 7, "num_conv_pos_embeddings": 16,
    "feat_extract_norm": "layer", "do_stable_layer_norm": True,
    "layerdrop": 0.0,  # igual que XLS-R en el modelo; si no, una capa puede quedar sin gradiente
}


def test_aasist_l_tiene_los_parametros_del_paper():
    model = build_model(load_config("configs/aasist_l.yaml", "laptop"))
    assert sum(p.numel() for p in model.parameters()) == 85_306  # AASIST-L oficial


def test_aasist_l_forma_de_salida_y_gradiente():
    model = build_model(load_config("configs/aasist_l.yaml", "laptop"))
    logits = model(torch.randn(2, NUM_SAMPLES))
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert model.encoder[0].conv1.weight.grad is not None
    assert model.head.out_layer.weight.grad is not None


def test_xlsr_forma_de_salida():
    model = XLSRAASIST(ssl_config=TINY_SSL)
    assert model(torch.randn(2, NUM_SAMPLES)).shape == (2, 2)


def test_xlsr_congelado_solo_entrena_la_cabeza():
    model = XLSRAASIST(ssl_config=TINY_SSL, freeze_ssl=True)
    assert not any(p.requires_grad for p in model.ssl.parameters())
    groups = model.param_groups({"lr_head": 1e-4, "lr_ssl": 1e-6})
    assert len(groups) == 1 and groups[0]["lr"] == 1e-4

    model.train()
    assert not model.ssl.training      # XLS-R sigue en eval aunque el modelo entrene
    assert model.head.training
    model(torch.randn(2, NUM_SAMPLES)).sum().backward()
    assert all(p.grad is None for p in model.ssl.parameters())
    assert model.LL.weight.grad is not None


def test_xlsr_fine_tuning_usa_dos_lr():
    model = XLSRAASIST(ssl_config=TINY_SSL, freeze_ssl=False)
    groups = model.param_groups({"lr_head": 1e-4, "lr_ssl": 1e-6})
    assert [g["lr"] for g in groups] == [1e-6, 1e-4]
    n_ssl = sum(p.numel() for p in model.ssl.parameters())
    assert sum(p.numel() for p in groups[0]["params"]) == n_ssl
    model.train()
    assert model.ssl.training
    model(torch.randn(2, NUM_SAMPLES)).sum().backward()
    assert model.ssl.encoder.layers[0].attention.q_proj.weight.grad is not None


def test_overrides_de_maquina_no_se_aplican_a_c0():
    c1 = load_config("configs/xlsr_aasist.yaml", "laptop")
    c0 = load_config("configs/xlsr_aasist_frozen.yaml", "laptop")
    assert (c1["train"]["batch_size"], c1["train"]["grad_accum"]) == (4, 4)
    assert (c0["train"]["batch_size"], c0["train"]["grad_accum"]) == (16, 1)
    assert c0["model_args"]["freeze_ssl"] and not c0["model_args"]["gradient_checkpointing"]
    for machine in ("laptop", "kaggle"):
        cfg = load_config("configs/xlsr_aasist.yaml", machine)
        assert cfg["train"]["batch_size"] * cfg["train"]["grad_accum"] == 16


@pytest.mark.parametrize("freeze", [True, False])
def test_train_py_entrena_xlsr(tmp_path, monkeypatch, freeze):
    cfg = load_yaml("configs/xlsr_aasist_frozen.yaml")
    cfg["model_args"] = {"freeze_ssl": freeze, "ssl_config": TINY_SSL}
    cfg["train"].update(batch_size=4, lr_ssl=1e-6)
    path = tmp_path / "xlsr.yaml"
    path.write_text(yaml.safe_dump(cfg))

    original = train.load_config
    monkeypatch.setattr(train, "load_config", lambda p, m: {**original(p, m),
                                                            "output_root": str(tmp_path)})
    monkeypatch.setattr(sys, "argv", [
        "prog", "--config", str(path), "--fake-data", "--device", "cpu", "--epochs", "1",
        "--limit-train-batches", "2", "--limit-dev-batches", "2", "--name", "xlsr"])
    train.main()
    assert (tmp_path / "xlsr" / "best.pt").exists()

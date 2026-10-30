"""Carga y combinación de configs: config del modelo + config de la máquina."""

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MACHINES_DIR = REPO_ROOT / "configs" / "machines"


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_machine(cfg: dict, machine: str) -> dict:
    """Combina un config de modelo con configs/machines/<machine>.yaml.

    Las claves de la máquina (device, precision, dataset_roots...) quedan en la raíz, y
    `overrides.<modelo>` se combina sección por sección (train, model_args...).
    """
    cfg = copy.deepcopy(cfg)
    machine_cfg = load_yaml(MACHINES_DIR / f"{machine}.yaml")
    overrides = machine_cfg.pop("overrides", None) or {}
    _deep_update(cfg, machine_cfg)
    _deep_update(cfg, overrides.get(cfg["model"]) or {})
    return cfg


def load_config(config_path: str | Path, machine: str) -> dict:
    cfg = load_yaml(config_path)
    cfg["config_path"] = str(Path(config_path))
    return apply_machine(cfg, machine)


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

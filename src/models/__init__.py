"""Registro de modelos. Todos reciben la forma de onda [B, 64600] y devuelven logits [B, 2]
(columna 1 = sintética).

Un modelo puede definir `param_groups(train_cfg) -> list[dict]` para usar tasas de aprendizaje
distintas por parte (p. ej. XLS-R con lr 1e-6 y la cabeza con 1e-4).
"""

from torch import nn


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]
    args = cfg.get("model_args") or {}
    if name == "debug":
        from src.models.debug import DebugCNN
        return DebugCNN(**args)
    if name == "lfcc_cnn":
        from src.models.lfcc_cnn import LFCCCNN
        return LFCCCNN(cfg)
    if name == "aasist_l":
        from src.models.aasist import AASIST
        return AASIST(**args)
    if name == "xlsr_aasist":
        from src.models.xlsr_aasist import XLSRAASIST
        return XLSRAASIST(**args)
    raise ValueError(f"Modelo desconocido: {name}")

"""Valor USD/mes en splits de prueba con los umbrales que eligió el modelo de valor en dev.

Los umbrales se fijan en dev (results/value/<modelo>_thresholds.json) y aquí solo se aplican,
así que mide cuánto valor se conserva fuera de dev: ataques no vistos (asv19_eval) y canal
telefónico (asv19_eval_codec, de scripts/precompute_codecs.py).

Uso:
    uv run python -m scripts.value_on_eval --models C0 C2 --splits asv19_eval asv19_eval_codec
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import REPO_ROOT
from src.value.value_model import DEFAULT_PARAMS, rates_at_threshold, value_at_thresholds

RESULTS = REPO_ROOT / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--splits", nargs="+", default=["asv19_eval", "asv19_eval_codec"])
    p.add_argument("--out", default=RESULTS / "value" / "valor_en_eval.csv")
    args = p.parse_args()

    rows = []
    for model in args.models:
        th = json.loads((RESULTS / "value" / f"{model}_thresholds.json").read_text())
        params = {**DEFAULT_PARAMS, "c_comp": th["costo_computo"] / DEFAULT_PARAMS["N"]}
        for split in args.splits:
            s = pd.read_csv(RESULTS / "scores" / f"{model}__{split}.csv")
            tpr1, fpr1 = rates_at_threshold(s.score.values, s.label.values, th["t1"])
            tpr2, fpr2 = rates_at_threshold(s.score.values, s.label.values, th["t2"])
            v = value_at_thresholds(tpr1, tpr2, fpr1, fpr2, params)
            rows.append({"modelo": model, "split": split, "t1": th["t1"], "t2": th["t2"],
                         "tpr_t1": tpr1, "fpr_t1": fpr1, "tpr_t2": tpr2, "fpr_t2": fpr2, **v})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df[["modelo", "split", "tpr_t2", "fpr_t2", "costo_alertas", "valor_usd_mes"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}" if x < 1 else f"{x:,.0f}"))
    print(f"Guardado: {args.out}")


if __name__ == "__main__":
    main()

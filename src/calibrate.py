"""Calibración de Platt: convierte el score del modelo en una P(sintética) interpretable.

El score crudo (softmax) refleja la proporción de clases de ASVspoof (aprox. 90 % sintético),
no la de un centro de contacto (0.1 % a 1 %). Se ajusta una regresión logística sobre el
logit del score en dev, con clases balanceadas, para obtener un log-likelihood ratio:

    llr = a · logit(score) + b
    P(sintética | prevalencia p) = sigmoid(llr + logit(p))

Así la misma calibración sirve para mostrar la probabilidad con la prevalencia real en el
panel. La transformación es monótona: no cambia EER, TPR ni FPR, solo la escala, así que los
umbrales t1 y t2 se pueden expresar sobre el score o sobre la probabilidad.

Uso:
    uv run python -m src.calibrate --scores results/scores/C1__asv19_dev.csv
    # -> runs/C1/calibration.json (o --out)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import REPO_ROOT

EPS = 1e-7


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def fit(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Ajusta a y b con clases balanceadas (prior 50/50)."""
    lr = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000)
    lr.fit(logit(scores).reshape(-1, 1), labels)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def llr(scores: np.ndarray, params: dict) -> np.ndarray:
    return params["a"] * logit(scores) + params["b"]


def calibrated_probability(scores: np.ndarray, params: dict, prior: float = 0.5) -> np.ndarray:
    """P(sintética) para una prevalencia `prior` de ataques."""
    return sigmoid(llr(scores, params) + logit(prior))


def brier(prob: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((prob - labels) ** 2, weights=weights))


def ece(prob: np.ndarray, labels: np.ndarray, weights: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error con pesos (para evaluarlo con clases balanceadas)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    total, err = weights.sum(), 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            w = weights[mask]
            err += w.sum() / total * abs(np.average(prob[mask], weights=w)
                                         - np.average(labels[mask], weights=w))
    return float(err)


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return np.where(labels == 1, 0.5 / n_pos, 0.5 / n_neg)


def main() -> None:
    p = argparse.ArgumentParser(description="Calibración de Platt sobre los scores de dev.")
    p.add_argument("--scores", required=True, help="results/scores/<run>__<split dev>.csv")
    p.add_argument("--out", help="por defecto runs/<run>/calibration.json")
    args = p.parse_args()

    path = Path(args.scores)
    df = pd.read_csv(path)
    scores, labels = df.score.to_numpy(), df.label.to_numpy()
    params = fit(scores, labels)

    w = balanced_weights(labels)
    raw, cal = np.clip(scores, 0, 1), calibrated_probability(scores, params)
    params.update({
        "fuente": str(path), "n": len(df),
        "brier_crudo": brier(raw, labels, w), "brier_calibrado": brier(cal, labels, w),
        "ece_crudo": ece(raw, labels, w), "ece_calibrado": ece(cal, labels, w),
    })

    run = path.stem.split("__")[0]
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / run / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"a = {params['a']:.4f}, b = {params['b']:.4f}  (n = {params['n']})")
    print(f"Brier balanceado: {params['brier_crudo']:.4f} -> {params['brier_calibrado']:.4f} | "
          f"ECE: {params['ece_crudo']:.4f} -> {params['ece_calibrado']:.4f}")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()

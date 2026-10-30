"""Métricas de detección. Convención del proyecto: score = P(sintética), label 1 = sintética.

min t-DCF sigue la implementación oficial de ASVspoof 2019 (la misma que usa clovaai/aasist),
que espera scores con la convención contraria (más alto = genuina); aquí se invierte el score
solo para ese cálculo.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Modelo de costos oficial de ASVspoof 2019 para t-DCF.
TDCF_COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,
    "Pnon": (1 - 0.05) * 0.01,
    "Cmiss_asv": 1,
    "Cfa_asv": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}


def _has_both_classes(labels: np.ndarray) -> bool:
    return np.unique(labels).size == 2


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """EER y su umbral. Devuelve (nan, nan) si falta una de las dos clases."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if not _has_both_classes(labels):
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if not _has_both_classes(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """IC por bootstrap estratificado (remuestrea dentro de cada clase para mantener la proporción)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if n <= 0 or not _has_both_classes(labels):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_by_class = [np.flatnonzero(labels == c) for c in (0, 1)]
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in idx_by_class])
        values.append(metric(scores[idx], labels[idx]))
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --- min t-DCF (ASVspoof 2019) ---------------------------------------------------------------

def _det_curve(target: np.ndarray, nontarget: np.ndarray):
    """FRR y FAR para todos los umbrales; target = clase con scores altos."""
    n_scores = target.size + nontarget.size
    all_scores = np.concatenate((target, nontarget))
    labels = np.concatenate((np.ones(target.size), np.zeros(nontarget.size)))
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate(([0.0], tar_trial_sums / target.size))
    far = np.concatenate(([1.0], nontarget_trial_sums / nontarget.size))
    thresholds = np.concatenate(([all_scores[order[0]] - 0.001], all_scores[order]))
    return frr, far, thresholds


def _eer_threshold(target: np.ndarray, nontarget: np.ndarray) -> float:
    frr, far, thresholds = _det_curve(target, nontarget)
    return float(thresholds[np.argmin(np.abs(frr - far))])


def load_asv_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lee un archivo de scores ASV de ASVspoof 2019: (claves, scores).

    Cada línea trae una clave target / nontarget / spoof y el score ASV al final.
    """
    keys, scores = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            key = next((p for p in parts if p in ("target", "nontarget", "spoof")), None)
            if key is None:
                continue
            keys.append(key)
            scores.append(float(parts[-1]))
    return np.array(keys), np.array(scores, dtype=float)


def compute_min_tdcf(
    scores: np.ndarray,
    labels: np.ndarray,
    asv_keys: np.ndarray,
    asv_scores: np.ndarray,
    cost_model: dict = TDCF_COST_MODEL,
) -> float:
    """min t-DCF normalizado (formulación de ASVspoof 2019)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    # Convención oficial: score alto = genuina.
    cm = 1.0 - scores
    bona_cm, spoof_cm = cm[labels == 0], cm[labels == 1]

    tar = asv_scores[asv_keys == "target"]
    non = asv_scores[asv_keys == "nontarget"]
    spoof_asv = asv_scores[asv_keys == "spoof"]
    asv_threshold = _eer_threshold(tar, non)
    p_fa_asv = np.mean(non >= asv_threshold)
    p_miss_asv = np.mean(tar < asv_threshold)
    p_miss_spoof_asv = np.mean(spoof_asv < asv_threshold)

    p_miss_cm, p_fa_cm, _ = _det_curve(bona_cm, spoof_cm)
    c = cost_model
    c1 = c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * p_miss_asv) - c["Pnon"] * c["Cfa_asv"] * p_fa_asv
    c2 = c["Cfa_cm"] * c["Pspoof"] * (1 - p_miss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise ValueError("Constantes de t-DCF negativas: revisar los scores ASV.")
    tdcf = c1 * p_miss_cm + c2 * p_fa_cm
    return float(np.min(tdcf / min(c1, c2)))


# --- Desgloses ----------------------------------------------------------------------------------

def eer_by_attack(df: pd.DataFrame) -> pd.DataFrame:
    """EER de cada ataque contra todas las genuinas. Requiere columnas score, label, attack."""
    bona = df[df.label == 0]
    rows = []
    for attack, spoof in df[df.label == 1].groupby("attack"):
        sub = pd.concat([bona, spoof])
        rows.append({"grupo": attack, "n_genuinas": len(bona), "n_sinteticas": len(spoof),
                     "eer": compute_eer(sub.score.values, sub.label.values)[0]})
    return pd.DataFrame(rows)


def eer_by_group(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """EER dentro de cada valor de `column` (p. ej. codec o language), si tiene ambas clases."""
    rows = []
    for value, sub in df.groupby(column):
        rows.append({"grupo": value, "n_genuinas": int((sub.label == 0).sum()),
                     "n_sinteticas": int((sub.label == 1).sum()),
                     "eer": compute_eer(sub.score.values, sub.label.values)[0]})
    return pd.DataFrame(rows)

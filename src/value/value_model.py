"""Modelo de valor de negocio (USD/mes) frente a "sin sistema".

Usa tasas TPR/FPR (no conteos del dataset) porque ASVspoof tiene ~90 % spoof y en un
centro de contacto la prevalencia es 0.1 %–1 %. Fórmula y parámetros base: PLAN.md §7.

Uso:
    uv run python -m src.value.value_model \\
        --scores results/scores/C0__asv19_dev.csv results/scores/C1__asv19_dev.csv \\
        --names C0 C1 --out results/value
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import REPO_ROOT

# Parámetros base documentados en PLAN.md §7 (fuentes en el escrito 4.5).
DEFAULT_PARAMS = {
    "N": 500_000,          # llamadas/mes (banco mediano en Centroamérica)
    "p": 0.002,            # prevalencia de ataques con voz sintética (0.2 %)
    "L": 2_500.0,          # pérdida promedio por fraude exitoso (USD)
    "e_v": 0.70,           # eficacia de la verificación extra
    "e_a": 0.95,           # eficacia del analista de fraude
    "c_v": 2.0,            # costo de una verificación adicional (USD)
    "c_a": 12.0,           # costo de transferir a un analista (USD)
    "c_comp": 0.0,         # costo de cómputo por llamada (se sobreescribe con latency.csv)
}


@dataclass
class ValueResult:
    name: str
    t1: float
    t2: float
    tpr_t1: float
    tpr_t2: float
    fpr_t1: float
    fpr_t2: float
    valor_usd_mes: float
    fraude_evitado: float
    costo_alertas: float
    costo_computo: float
    valor_verificar_todos: float
    eer: float


def rates_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """TPR y FPR con score ≥ threshold → positivo (sintética)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pred = scores >= threshold
    pos, neg = labels == 1, labels == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr


def value_at_thresholds(
    tpr_t1: float,
    tpr_t2: float,
    fpr_t1: float,
    fpr_t2: float,
    params: dict,
) -> dict[str, float]:
    """Valor neto mensual frente a 'sin sistema' (PLAN.md §7)."""
    N, p, L = params["N"], params["p"], params["L"]
    e_v, e_a = params["e_v"], params["e_a"]
    c_v, c_a, c_comp = params["c_v"], params["c_a"], params["c_comp"]

    # Garantiza t1 ≤ t2 en tasas: si el usuario invierte umbrales, se ordenan.
    if tpr_t1 < tpr_t2:
        tpr_t1, tpr_t2 = tpr_t2, tpr_t1
        fpr_t1, fpr_t2 = fpr_t2, fpr_t1

    fraude_evitado = N * p * L * (e_a * tpr_t2 + e_v * (tpr_t1 - tpr_t2))
    costo_alertas = (
        N * (1 - p) * (c_v * (fpr_t1 - fpr_t2) + c_a * fpr_t2)
        + N * p * (c_v * (tpr_t1 - tpr_t2) + c_a * tpr_t2)
    )
    costo_computo = N * c_comp
    valor = fraude_evitado - costo_alertas - costo_computo
    # Línea base "verificar a todos": TPR=1, FPR=1 con solo verificación (e_v, c_v).
    valor_todos = N * p * L * e_v - N * c_v - costo_computo
    return {
        "fraude_evitado": float(fraude_evitado),
        "costo_alertas": float(costo_alertas),
        "costo_computo": float(costo_computo),
        "valor_usd_mes": float(valor),
        "valor_verificar_todos": float(valor_todos),
    }


def load_c_comp(latency_csv: Path, model_key: str | None = None) -> float:
    """USD por llamada a partir de `usd_por_1000_llamadas` en results/latency.csv."""
    if not latency_csv.is_file():
        return 0.0
    df = pd.read_csv(latency_csv)
    if model_key and "modelo" in df.columns:
        hit = df[df.modelo.astype(str).str.contains(model_key, case=False, regex=False)]
        if not hit.empty:
            return float(hit.iloc[0]["usd_por_1000_llamadas"]) / 1000.0
    if "usd_por_1000_llamadas" in df.columns and not df.empty:
        return float(df["usd_por_1000_llamadas"].mean()) / 1000.0
    return 0.0


def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def grid_search_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    params: dict,
    n_grid: int = 51,
) -> ValueResult:
    """Elige t1 < t2 que maximizan valor(t1, t2) sobre una grilla de cuantiles."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    # Umbrales candidatos: cuantiles del score (evita evaluar N² puntos crudos).
    qs = np.linspace(0.01, 0.99, n_grid)
    candidates = np.unique(np.quantile(scores, qs))
    best = None
    best_val = -np.inf
    for i, t2 in enumerate(candidates):
        for t1 in candidates[: i + 1]:
            if t1 > t2:
                continue
            tpr1, fpr1 = rates_at_threshold(scores, labels, t1)
            tpr2, fpr2 = rates_at_threshold(scores, labels, t2)
            out = value_at_thresholds(tpr1, tpr2, fpr1, fpr2, params)
            if out["valor_usd_mes"] > best_val:
                best_val = out["valor_usd_mes"]
                best = ValueResult(
                    name="",
                    t1=float(t1),
                    t2=float(t2),
                    tpr_t1=tpr1,
                    tpr_t2=tpr2,
                    fpr_t1=fpr1,
                    fpr_t2=fpr2,
                    valor_usd_mes=out["valor_usd_mes"],
                    fraude_evitado=out["fraude_evitado"],
                    costo_alertas=out["costo_alertas"],
                    costo_computo=out["costo_computo"],
                    valor_verificar_todos=out["valor_verificar_todos"],
                    eer=eer_from_scores(scores, labels),
                )
    assert best is not None
    return best


def value_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    params: dict,
    n_points: int = 101,
) -> pd.DataFrame:
    """Curva de valor vs umbral único t (= t1 = t2): útil para la figura del escrito."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    thresholds = np.linspace(0, 1, n_points)
    rows = []
    for t in thresholds:
        tpr, fpr = rates_at_threshold(scores, labels, t)
        out = value_at_thresholds(tpr, tpr, fpr, fpr, params)
        rows.append({"threshold": t, "tpr": tpr, "fpr": fpr, **out})
    return pd.DataFrame(rows)


def evaluate_model(
    scores_csv: Path,
    name: str,
    params: dict,
    n_grid: int = 51,
) -> tuple[ValueResult, pd.DataFrame]:
    df = pd.read_csv(scores_csv)
    scores = df["score"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    result = grid_search_thresholds(scores, labels, params, n_grid=n_grid)
    result.name = name
    curve = value_curve(scores, labels, params)
    curve.insert(0, "model", name)
    return result, curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modelo de valor USD/mes y umbrales óptimos.")
    p.add_argument("--scores", nargs="+", required=True, help="CSV con columnas score,label")
    p.add_argument("--names", nargs="+", help="nombres de cada corrida (default: stem del CSV)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "value")
    p.add_argument("--latency", type=Path, default=REPO_ROOT / "results" / "latency.csv")
    p.add_argument("--c-comp", type=float, help="sobrescribe costo de cómputo por llamada")
    p.add_argument("--N", type=float, default=DEFAULT_PARAMS["N"])
    p.add_argument("--p", type=float, default=DEFAULT_PARAMS["p"])
    p.add_argument("--L", type=float, default=DEFAULT_PARAMS["L"])
    p.add_argument("--grid", type=int, default=51)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = args.names or [Path(s).stem.split("__")[0] for s in args.scores]
    if len(names) != len(args.scores):
        raise SystemExit("--names debe tener la misma longitud que --scores")

    params = {**DEFAULT_PARAMS, "N": args.N, "p": args.p, "L": args.L}
    if args.c_comp is not None:
        params["c_comp"] = args.c_comp

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_curves = []
    for path, name in zip(args.scores, names):
        local = dict(params)
        if args.c_comp is None:
            key = "xlsr" if name.upper().startswith("C") else (
                "aasist" if name.upper().startswith("B") else "lfcc"
            )
            local["c_comp"] = load_c_comp(args.latency, key)
        result, curve = evaluate_model(Path(path), name, local, n_grid=args.grid)
        summary_rows.append(asdict(result))
        all_curves.append(curve)
        (args.out / f"{name}_thresholds.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8",
        )
        print(
            f"{name}: valor=${result.valor_usd_mes:,.0f}/mes  "
            f"t1={result.t1:.4f} t2={result.t2:.4f}  EER={result.eer:.4f}"
        )

    summary_path = args.out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    pd.concat(all_curves, ignore_index=True).to_csv(args.out / "curves.csv", index=False)
    print(f"Guardado: {summary_path}")


if __name__ == "__main__":
    main()

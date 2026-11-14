"""Análisis de sensibilidad del modelo de valor (tornado ±20/50 % y heatmap p × L).

Uso:
    uv run python -m src.value.sensitivity \\
        --scores results/scores/C0__asv19_dev.csv --name C0 --out results/value
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPO_ROOT
from src.value.value_model import (
    DEFAULT_PARAMS,
    evaluate_model,
    load_c_comp,
    rates_at_threshold,
    value_at_thresholds,
)

# Parámetros que se varían en el tornado (c_comp se toma de latency.csv).
TORNADO_KEYS = ("N", "p", "L", "e_v", "e_a", "c_v", "c_a", "c_comp")


def tornado(
    scores: np.ndarray,
    labels: np.ndarray,
    base_params: dict,
    t1: float,
    t2: float,
    deltas: tuple[float, ...] = (0.2, 0.5),
) -> pd.DataFrame:
    """Para cada parámetro, valor con ±delta relativo manteniendo t1/t2 fijos."""
    tpr1, fpr1 = rates_at_threshold(scores, labels, t1)
    tpr2, fpr2 = rates_at_threshold(scores, labels, t2)
    base = value_at_thresholds(tpr1, tpr2, fpr1, fpr2, base_params)["valor_usd_mes"]
    rows = []
    for key in TORNADO_KEYS:
        for d in deltas:
            for sign, label in ((-1, f"-{int(d * 100)}%"), (1, f"+{int(d * 100)}%")):
                params = dict(base_params)
                params[key] = base_params[key] * (1 + sign * d)
                val = value_at_thresholds(tpr1, tpr2, fpr1, fpr2, params)["valor_usd_mes"]
                rows.append({
                    "param": key,
                    "delta": label,
                    "delta_frac": sign * d,
                    "valor_usd_mes": val,
                    "delta_vs_base": val - base,
                    "base_valor": base,
                })
    return pd.DataFrame(rows)


def heatmap_prevalence_loss(
    scores: np.ndarray,
    labels: np.ndarray,
    base_params: dict,
    t1: float,
    t2: float,
    p_grid: np.ndarray | None = None,
    l_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Valor en una malla de prevalencia × pérdida promedio."""
    p_grid = p_grid if p_grid is not None else np.array([0.0005, 0.001, 0.002, 0.005, 0.01])
    l_grid = l_grid if l_grid is not None else np.array([1000, 2500, 5000, 10000])
    tpr1, fpr1 = rates_at_threshold(scores, labels, t1)
    tpr2, fpr2 = rates_at_threshold(scores, labels, t2)
    rows = []
    for p in p_grid:
        for L in l_grid:
            params = {**base_params, "p": float(p), "L": float(L)}
            val = value_at_thresholds(tpr1, tpr2, fpr1, fpr2, params)["valor_usd_mes"]
            rows.append({"p": float(p), "L": float(L), "valor_usd_mes": val})
    return pd.DataFrame(rows)


def plot_tornado(df: pd.DataFrame, out: Path, title: str) -> None:
    """Diagrama tornado con la variación ±50 % (la de mayor rango)."""
    sub = df[df.delta.isin(["-50%", "+50%"])].copy()
    # Rango absoluto por parámetro para ordenar.
    spans = (
        sub.groupby("param")["delta_vs_base"]
        .agg(lambda s: s.max() - s.min())
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(spans))
    for i, param in enumerate(spans.index):
        neg = sub[(sub.param == param) & (sub.delta == "-50%")]["delta_vs_base"].iloc[0]
        pos = sub[(sub.param == param) & (sub.delta == "+50%")]["delta_vs_base"].iloc[0]
        ax.barh(i, neg, color="#4C78A8", height=0.6)
        ax.barh(i, pos, color="#F58518", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(list(spans.index))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ valor USD/mes vs base (±50 %)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out: Path, title: str) -> None:
    pivot = df.pivot(index="p", columns="L", values="valor_usd_mes")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c):,}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{p * 100:.2f}%" for p in pivot.index])
    ax.set_xlabel("L (USD por fraude)")
    ax.set_ylabel("Prevalencia p")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="USD/mes")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensibilidad del modelo de valor.")
    p.add_argument("--scores", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "value")
    p.add_argument("--latency", type=Path, default=REPO_ROOT / "results" / "latency.csv")
    p.add_argument("--thresholds", type=Path, help="JSON con t1/t2; si no, se buscan")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name or Path(args.scores).stem.split("__")[0]
    params = dict(DEFAULT_PARAMS)
    key = "xlsr" if name.upper().startswith("C") else (
        "aasist" if name.upper().startswith("B") else "lfcc"
    )
    params["c_comp"] = load_c_comp(args.latency, key)

    df = pd.read_csv(args.scores)
    scores, labels = df["score"].to_numpy(float), df["label"].to_numpy(int)

    if args.thresholds and args.thresholds.is_file():
        thr = json.loads(args.thresholds.read_text())
        t1, t2 = thr["t1"], thr["t2"]
    else:
        result, _ = evaluate_model(Path(args.scores), name, params)
        t1, t2 = result.t1, result.t2

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    torn = tornado(scores, labels, params, t1, t2)
    torn.to_csv(args.out / f"{name}_tornado.csv", index=False)
    plot_tornado(torn, fig_dir / f"{name}_tornado.png", f"Tornado ±50 % — {name}")

    heat = heatmap_prevalence_loss(scores, labels, params, t1, t2)
    heat.to_csv(args.out / f"{name}_heatmap.csv", index=False)
    plot_heatmap(heat, fig_dir / f"{name}_heatmap_pL.png", f"Valor vs p × L — {name}")
    print(f"Sensibilidad de {name}: t1={t1:.4f} t2={t2:.4f} → {args.out}")


if __name__ == "__main__":
    main()

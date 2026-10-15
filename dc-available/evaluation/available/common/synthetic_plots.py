"""Plots for the synthetic evaluation — reads the JSON outputs, writes PNGs.

Inputs  (evaluation/results/synthetic/):
  retrieval_metrics.json           Hit@k + MRR per leg (plot 1)
  vector.json / graph.json / hybrid.json   per-leg records (plot 2)
  ragas_scores.json                per-question RAGAS metrics (plot 4)

Outputs (evaluation/results/synthetic/plots/):
  1_retrieval_metrics.png   Hit@1/3/10 + MRR per leg
  2_identified.png          golden identified vs not (top-10) per leg, grouped bars
  4_ragas_summary.png       4 RAGAS metrics (hybrid)

Run from the project root (dc-available):
  poetry run python -m evaluation.available.common.synthetic_plots
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("evaluation/results/synthetic")
OUT = RES / "plots"

LEGS = ["vector", "graph", "hybrid"]
LEG_LABEL = {"vector": "Vector", "graph": "Graph", "hybrid": "Hybrid"}
LEG_COLOR = {"vector": "#2980b9", "graph": "#e67e22", "hybrid": "#27ae60"}

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABEL = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
MET_COLOR = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

GREEN, RED = "#2ecc71", "#e74c3c"
TOP = 10


def _load():
    met = json.loads((RES / "retrieval_metrics.json").read_text(encoding="utf-8"))
    rec = []
    for leg in LEGS:
        rec += json.loads((RES / f"{leg}.json").read_text(encoding="utf-8"))
    rg = pd.read_json(RES / "ragas_scores.json")
    return met, rec, rg


def _identified(r) -> bool:
    gr = r.get("golden_rank")
    return gr is not None and gr <= TOP


# --- 1: retrieval metrics per leg -------------------------------------------
def plot1_retrieval_metrics(met) -> None:
    keys = ["hit@1", "hit@3", "hit@10", "mrr"]
    labels = ["Hit@1", "Hit@3", "Hit@10", "MRR"]
    x = range(len(keys))
    w = 0.25
    plt.figure(figsize=(9, 5))
    for i, leg in enumerate(LEGS):
        offs = [xi + (i - 1) * w for xi in x]
        vals = [met[leg][k] for k in keys]
        bars = plt.bar(offs, vals, w, label=LEG_LABEL[leg], color=LEG_COLOR[leg])
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.05)
    plt.ylabel("score (0–1)")
    plt.title("Retrieval metrics per leg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "1_retrieval_metrics.png", dpi=150)
    plt.close()


# --- 2: identified vs not-identified per leg (grouped bars) ------------------
def plot2_identified(rec) -> None:
    ident = [sum(1 for r in rec if r["retrieval_type"] == leg and _identified(r)) for leg in LEGS]
    tot = [sum(1 for r in rec if r["retrieval_type"] == leg) for leg in LEGS]
    miss = [t - i for t, i in zip(tot, ident)]
    x = range(len(LEGS))
    w = 0.38
    plt.figure(figsize=(8, 5))
    b1 = plt.bar([xi - w / 2 for xi in x], ident, w, label="Identified", color=GREEN)
    b2 = plt.bar([xi + w / 2 for xi in x], miss, w, label="Not identified", color=RED)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, int(h),
                     ha="center", va="bottom", fontsize=9)
    plt.xticks(list(x), [LEG_LABEL[l] for l in LEGS])
    plt.ylabel(f"questions (of {tot[0]})")
    plt.title(f"Golden document identified in top-{TOP} per leg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "2_identified.png", dpi=150)
    plt.close()


# --- 4: RAGAS summary --------------------------------------------------------
def plot4_ragas_summary(rg) -> None:
    vals = [float(rg[m].mean(skipna=True)) for m in METRICS]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(METRIC_LABEL, vals, 0.5, color=MET_COLOR)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 1.15)
    plt.ylabel("score (0–1)")
    plt.title("RAGAS generation quality — hybrid")
    plt.tight_layout()
    plt.savefig(OUT / "4_ragas_summary.png", dpi=150)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    met, rec, rg = _load()
    plot1_retrieval_metrics(met)
    plot2_identified(rec)
    plot4_ragas_summary(rg)
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()

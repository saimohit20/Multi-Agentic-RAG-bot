"""
Build the two evaluation plots from the result JSON files.

    .venv/Scripts/python.exe evaluation/retrieval_eval/src/make_plots.py

Plot 1 (identification.png):    x=topics, y=# questions, groups = identified / not identified
Plot 2 (retrieval_metrics.png): x=topics, y=score [0,1], groups = hit@k / MRR
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np

import paths_config as cfg
import eval_metrics


def _load(path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _grouped_bar(ax, topics, series, ylabel, title, value_fmt):
    x = np.arange(len(topics))
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, (label, values) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label)
        for bar, val in zip(bars, values):
            ax.annotate(
                value_fmt(val),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(topics)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def main() -> None:
    cfg.ensure_dirs()

    sources = [
        ("Grownfield", cfg.GROWNFIELD_RESULTS),
        ("Hydraulics", cfg.HYDRAULICS_RESULTS),
        ("Synthetic", cfg.SYNTHETIC_RESULTS),
    ]
    topics: List[str] = []
    metrics: Dict[str, Dict[str, Any]] = {}
    for name, path in sources:
        if path.exists():
            metrics[name] = eval_metrics.compute(_load(path))
            topics.append(name)
        else:
            print(f"[Plots] Skipping {name}: {path} not found (run its runner first).")

    if not topics:
        print("[Plots] No result files found. Run run_grownfield.py / run_hydraulics.py first.")
        return

    hit_k = metrics[topics[0]]["hit_k"]

    print("\n=== Retrieval summary ===")
    for t in topics:
        print(eval_metrics.format_summary(t, metrics[t]))

    # --- Plot 1: identification counts --------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    _grouped_bar(
        ax1, topics,
        series={
            "Identified": [metrics[t]["identified"] for t in topics],
            "Not identified": [metrics[t]["not_identified"] for t in topics],
        },
        ylabel="No. of questions",
        title=f"Source identification (golden doc in top-{cfg.TOP_K})",
        value_fmt=lambda v: f"{int(v)}",
    )
    fig1.tight_layout()
    out1 = cfg.PLOTS_DIR / "identification.png"
    fig1.savefig(out1, dpi=150)
    print(f"\nSaved -> {out1}")

    # --- Plot 2: retrieval metrics ------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    _grouped_bar(
        ax2, topics,
        series={
            f"hit@{hit_k}": [metrics[t][f"hit@{hit_k}"] for t in topics],
            "MRR": [metrics[t]["mrr"] for t in topics],
        },
        ylabel="Score",
        title="Retrieval metrics",
        value_fmt=lambda v: f"{v:.2f}",
    )
    ax2.set_ylim(0, 1)
    fig2.tight_layout()
    out2 = cfg.PLOTS_DIR / "retrieval_metrics.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved -> {out2}")


if __name__ == "__main__":
    main()

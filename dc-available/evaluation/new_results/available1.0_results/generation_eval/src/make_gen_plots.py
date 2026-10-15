"""
Build two RAGAS plots from ragas_summary.json.

Plot 1 (ragas_metrics.png): grouped bar chart — 4 metrics × 2 topics, y-axis 0-1
Plot 2 (ragas_radar.png):   radar / spider chart — 4 axes × 2 topics

Run:
    .venv/Scripts/python.exe evaluation/generation_eval/src/make_gen_plots.py
"""
from __future__ import annotations

import json
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gen_paths_config as cfg

_COLORS = ["#4C72B0", "#DD8452"]

_METRIC_LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer\nRelevancy",
    "context_precision":  "Context\nPrecision",
    "context_recall":     "Context\nRecall",
}


def _load_summary() -> Dict[str, Dict[str, float]]:
    with open(cfg.RAGAS_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_bar(summary: Dict[str, Dict[str, float]]) -> None:
    topics = [t for t in summary if summary[t]]
    metrics = cfg.RAGAS_METRICS
    x_labels = [_METRIC_LABELS[m] for m in metrics]
    x = np.arange(len(metrics))
    n = len(topics)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, topic in enumerate(topics):
        vals = [summary[topic].get(m, 0.0) for m in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=topic.capitalize(),
            color=_COLORS[i % len(_COLORS)],
        )
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("RAGAS Metrics by Topic")
    ax.legend()
    fig.tight_layout()

    out = cfg.PLOTS_DIR / "ragas_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def _plot_radar(summary: Dict[str, Dict[str, float]]) -> None:
    topics = [t for t in summary if summary[t]]
    metrics = cfg.RAGAS_METRICS
    labels = [_METRIC_LABELS[m].replace("\n", " ") for m in metrics]
    N = len(metrics)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for i, topic in enumerate(topics):
        vals = [summary[topic].get(m, 0.0) for m in metrics]
        vals += vals[:1]
        color = _COLORS[i % len(_COLORS)]
        ax.plot(angles, vals, "o-", linewidth=2, label=topic.capitalize(), color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS Metrics Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    out = cfg.PLOTS_DIR / "ragas_radar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    cfg.ensure_dirs()

    if not cfg.RAGAS_SUMMARY.exists():
        print(f"[Plots] Not found: {cfg.RAGAS_SUMMARY}  (run run_ragas.py first)")
        return

    summary = _load_summary()

    print("\n=== RAGAS summary ===")
    for topic, scores in summary.items():
        if scores:
            parts = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
            print(f"  {topic}: {parts}")

    _plot_bar(summary)
    _plot_radar(summary)


if __name__ == "__main__":
    main()

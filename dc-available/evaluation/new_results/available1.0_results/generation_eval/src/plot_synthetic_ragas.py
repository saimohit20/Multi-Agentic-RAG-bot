"""
Plot the RAGAS scores for the SYNTHETIC dataset.

    .venv/Scripts/python.exe evaluation/generation_eval/src/plot_synthetic_ragas.py

Reads : evaluation/generation_eval/results/synthetic_ragas_summary.json
Writes: evaluation/generation_eval/results/plots/synthetic_ragas.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

import gen_paths_config as cfg

_METRIC_LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer\nRelevancy",
    "context_precision":  "Context\nPrecision",
    "context_recall":     "Context\nRecall",
}


def main() -> None:
    cfg.ensure_dirs()

    summary_path = cfg.ragas_summary_path("synthetic")
    if not summary_path.exists():
        print(f"[Plot] Not found: {summary_path}  (run run_ragas.py --topic synthetic first)")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    scores = summary.get("synthetic") or {}
    if not scores:
        print(f"[Plot] No 'synthetic' scores in {summary_path}.")
        return

    metrics = cfg.RAGAS_METRICS
    labels = [_METRIC_LABELS[m] for m in metrics]
    values = [scores.get(m, 0.0) for m in metrics]

    print("=== Synthetic RAGAS ===")
    for m in metrics:
        print(f"  {m}: {scores.get(m):.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#4C72B0", width=0.6)

    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 3),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Synthetic — RAGAS metrics")

    fig.tight_layout()
    out = cfg.PLOTS_DIR / "synthetic_ragas.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

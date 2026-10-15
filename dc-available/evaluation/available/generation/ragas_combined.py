"""Step 3 of RAGAS pipeline: cross-topic comparison plot.

Reads  evaluation/results/RAGAS/{grownfield,hydraulics}/summary.json
Writes evaluation/results/RAGAS/combined/
  summary.json             side-by-side aggregates
  plot_ragas_combined.png  grouped bar chart (4 metrics × 2 topics)

Standalone usage (from project root):
  poetry run python -m evaluation.available.generation.ragas_combined
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOPICS = [("grownfield", "Grownfield"), ("hydraulics", "Hydraulics")]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
RAGAS_DIR = Path("evaluation/results/RAGAS")
TOPIC_COLORS = {"grownfield": "#1f77b4", "hydraulics": "#ff7f0e"}


def _load_summary(topic: str, ragas_dir: Path) -> dict[str, float]:
    path = ragas_dir / topic / "summary.json"
    if not path.exists():
        sys.exit(f"Missing {path}  —  run ragas_eval.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main(ragas_dir: Path = RAGAS_DIR) -> None:
    summaries = {t: _load_summary(t, ragas_dir) for t, _ in TOPICS}

    out_dir = ragas_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    # print table
    header = "  ".join(f"{label:>12}" for _, label in TOPICS)
    print(f"\n{'Metric':24}  {header}")
    for m in METRIC_NAMES:
        vals = "  ".join(f"{summaries[t].get(m, float('nan')):>12.4f}" for t, _ in TOPICS)
        print(f"  {m:22}  {vals}")

    # grouped bar chart
    x = np.arange(len(METRIC_NAMES))
    n = len(TOPICS)
    w = 0.35
    plt.figure(figsize=(11, 6))
    for i, (topic, label) in enumerate(TOPICS):
        vals = [summaries[topic].get(m, 0.0) for m in METRIC_NAMES]
        offs = x + (i - (n - 1) / 2) * w
        bars = plt.bar(offs, vals, w, label=label, color=TOPIC_COLORS[topic])
        for bar, v in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
    plt.xticks(x, METRIC_LABELS)
    plt.ylim(0, 1.15)
    plt.ylabel("Score (0–1)")
    plt.title("RAGAS generation quality — Grownfield vs Hydraulics (hybrid leg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_ragas_combined.png", dpi=150)
    plt.close()

    print(f"\nWrote to {out_dir}:")
    print("  summary.json")
    print("  plot_ragas_combined.png")


if __name__ == "__main__":
    main()

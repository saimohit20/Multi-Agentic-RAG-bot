"""Combine the per-topic graph-engine results (E1) into one thesis view.

Pure read + plot (no RAG stack): reads the per-topic metrics.json written by
graph_ablation.py for grownfield + hydraulics and produces:

  combined.csv                 every (topic, engine) row + a question-weighted
                               "combined" row per engine (micro-average; n noted)
  plot_combined.png            two panels (MRR, Hit@10): dataset on x, engines as
                               grouped bars — shows the engine ranking per dataset

Datasets differ in difficulty (hydraulics is single-document → near-ceiling), so we
show them side by side rather than as one pooled bar; the weighted average is kept
in the CSV only, for completeness.

Usage:
  poetry run python -m evaluation.available.retrieval.graph_ablation_combine [results_dir]
Default results_dir: evaluation/new_results/graph_ablation/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Engines + per-topic question counts (for the weighted-average row).
ENGINES = ["entity_1hop", "entity_3hop", "ppr"]
ENGINE_LABELS = {"entity_1hop": "Entity 1-hop", "entity_3hop": "Entity 3-hop", "ppr": "PPR"}
TOPICS = [("grownfield", "Grownfield", 29), ("hydraulics", "Hydraulics", 59)]

# Wong colour-blind-safe palette (shared with plot_results.py / graph_ablation.py).
WONG = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "vermillion": "#D55E00", "grey": "#999999"}
ENGINE_COLOR = {"entity_1hop": WONG["orange"], "entity_3hop": WONG["blue"], "ppr": WONG["green"]}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def load(results_dir: Path) -> dict[str, dict]:
    """topic -> metrics dict (engine -> {hit@k, mrr})."""
    out = {}
    for topic, _, _ in TOPICS:
        p = results_dir / topic / "metrics.json"
        if not p.exists():
            raise SystemExit(f"[error] {p} not found — run graph_ablation for '{topic}' first.")
        out[topic] = json.loads(p.read_text(encoding="utf-8"))
    return out


def write_csv(data: dict, out: Path) -> None:
    keys = ["hit@1", "hit@5", "hit@10", "mrr"]
    lines = ["topic,engine," + ",".join(keys)]
    for topic, _, _ in TOPICS:
        for e in ENGINES:
            m = data[topic][e]
            lines.append(f"{topic},{e}," + ",".join(f"{m[k]:.4f}" for k in keys))
    # question-weighted ("combined") row per engine.
    total_n = sum(n for _, _, n in TOPICS)
    for e in ENGINES:
        row = []
        for k in keys:
            v = sum(n * data[t][e][k] for t, _, n in TOPICS) / total_n
            row.append(f"{v:.4f}")
        lines.append(f"combined(n={total_n}),{e}," + ",".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _panel(ax, data, metric, title) -> None:
    topics = [lbl for _, lbl, _ in TOPICS]
    x = np.arange(len(topics))
    w = 0.8 / len(ENGINES)
    for i, e in enumerate(ENGINES):
        vals = [data[t][e][metric] for t, _, _ in TOPICS]
        offs = x + (i - (len(ENGINES) - 1) / 2) * w
        bars = ax.bar(offs, vals, w, label=ENGINE_LABELS[e], color=ENGINE_COLOR[e])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(topics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(title)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)


def plot_combined(data: dict, out: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _panel(axes[0], data, "mrr", "MRR by engine")
    _panel(axes[1], data, "hit@10", "Hit@10 by engine")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(ENGINES),
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation/new_results/graph_ablation")
    out_dir = results_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(results_dir)
    write_csv(data, out_dir / "combined.csv")
    plot_combined(data, out_dir / "plot_combined.png")
    print(f"Wrote:\n  {out_dir/'combined.csv'}\n  {out_dir/'plot_combined.png'}")


if __name__ == "__main__":
    main()

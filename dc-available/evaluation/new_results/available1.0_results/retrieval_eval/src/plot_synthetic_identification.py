"""
Plot identified vs not-identified for the SYNTHETIC retrieval results.

"Identified" = the golden source document appears in the retrieved top-k
(golden_rank is not None).

    .venv/Scripts/python.exe evaluation/retrieval_eval/src/plot_synthetic_identification.py

Reads : evaluation/retrieval_eval/results/synthetic_results.json
Writes: evaluation/retrieval_eval/results/plots/synthetic_identification.png
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

import paths_config as cfg
import eval_metrics


def _load(path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    cfg.ensure_dirs()

    if not cfg.SYNTHETIC_RESULTS.exists():
        print(f"[Plot] Not found: {cfg.SYNTHETIC_RESULTS}  (run run_synthetic.py first)")
        return

    records = _load(cfg.SYNTHETIC_RESULTS)
    m = eval_metrics.compute(records)
    total = m["n"]
    identified = m["identified"]
    not_identified = m["not_identified"]

    print(eval_metrics.format_summary("Synthetic", m))

    labels = ["Identified", "Not identified"]
    values = [identified, not_identified]
    colors = ["#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    for bar, val in zip(bars, values):
        pct = (val / total * 100) if total else 0.0
        ax.annotate(
            f"{val}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 3),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("No. of questions")
    ax.set_ylim(0, total * 1.12)
    ax.set_title(f"Synthetic — source identification (top-{cfg.TOP_K})\n"
                 f"n = {total}")

    fig.tight_layout()
    out = cfg.PLOTS_DIR / "synthetic_identification.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

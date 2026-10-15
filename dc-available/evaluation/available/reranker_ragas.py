"""E3 generation quality: score the reranker-ablation answers with the SAME 4
RAGAS metrics used in E2 (faithfulness, answer_relevancy, context_precision,
context_recall). No re-generation — reuses the generated_answer already stored in
each config JSON (no_rerank / base / v2m3), so the generation metric stays
identical to E2 (no judge / answer_correctness here, by design).

Reuses evaluation.available.generation.ragas_eval.evaluate_topic for scoring, so
results are byte-identical to the E2 RAGAS path.

Reads   <results_dir>/{no_rerank,base,v2m3}.json      (reranker_ablation output)
Writes  <out_dir>/<config>/  scores.csv/json, summary.json, plot_ragas.png
        <out_dir>/combined_summary.csv / .json
        <out_dir>/plot_generation_ragas.png   grouped bars: metric x config

Usage (from project root):
  poetry run python -m evaluation.available.reranker_ragas [results_dir] [out_dir]
Default results_dir: evaluation/results/reranker_ablation/grownfield
Default out_dir:     <results_dir>/ragas
"""
from __future__ import annotations

# Opik off + silence noisy loggers (same guard as reranker_ablation.py).
import os

os.environ["OPIK_TRACK_DISABLE"] = "true"
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import logging

logging.getLogger("opik").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.ERROR)

import json
import sys
from pathlib import Path

# Drop the eval dirs so local evaluation/datasets can't shadow HF "datasets",
# but KEEP the project root (dc-available) so dc_available imports.
_here = Path(__file__).resolve().parent        # .../evaluation/available
_root = _here.parents[1]                        # .../dc-available
_drop = {str(_here), str(_here.parent)}         # available, evaluation
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _drop]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.available.generation.ragas_eval import (
    _build_ragas_llm_emb, evaluate_topic, METRIC_NAMES, METRIC_LABELS,
)

CONFIGS = ["no_rerank", "base", "v2m3"]
LABELS = {"no_rerank": "No reranker", "base": "bge-reranker-base", "v2m3": "bge-reranker-v2-m3"}

# Wong palette (canonical thesis style — matches plot_results.py).
WONG = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "grey": "#999999"}
CONFIG_COLOR = {"no_rerank": WONG["blue"], "base": WONG["orange"], "v2m3": WONG["green"]}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def plot_grouped(summaries: dict[str, dict], out: Path) -> None:
    set_style()
    cfgs = [c for c in CONFIGS if c in summaries]
    x = np.arange(len(METRIC_NAMES))
    w = 0.8 / max(1, len(cfgs))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, cfg in enumerate(cfgs):
        vals = [summaries[cfg].get(m, 0.0) for m in METRIC_NAMES]
        offs = x + (i - (len(cfgs) - 1) / 2) * w
        bars = ax.bar(offs, vals, w, label=LABELS[cfg], color=CONFIG_COLOR[cfg])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("RAGAS generation quality by reranker (hybrid leg)", pad=34)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, ncol=len(cfgs), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation/results/reranker_ablation/grownfield")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir / "ragas"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm, emb = _build_ragas_llm_emb()
    summaries: dict[str, dict] = {}
    for cfg in CONFIGS:
        in_file = results_dir / f"{cfg}.json"
        if not in_file.exists():
            print(f"[skip] {in_file} not found — run reranker_ablation first.")
            continue
        print(f"\n=== [ragas] {cfg} ===")
        # topic label only drives the per-config plot title; "grownfield" is fine.
        summaries[cfg] = evaluate_topic("grownfield", llm, emb, in_file=in_file, out_dir=out_dir / cfg)
        for m, v in summaries[cfg].items():
            print(f"    {m}: {v:.4f}")

    if not summaries:
        print("Nothing scored.")
        return

    # combined summary (one row per config) + grouped plot
    rows = [{"config": c, **summaries[c]} for c in CONFIGS if c in summaries]
    (out_dir / "combined_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    header = "config," + ",".join(METRIC_NAMES)
    lines = [header] + [
        f"{r['config']}," + ",".join(f"{r.get(m, 0.0):.4f}" for m in METRIC_NAMES) for r in rows
    ]
    (out_dir / "combined_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_grouped(summaries, out_dir / "plot_generation_ragas.png")
    print(f"\nWrote per-config scores + combined_summary.csv/json + "
          f"plot_generation_ragas.png to {out_dir}")


if __name__ == "__main__":
    main()

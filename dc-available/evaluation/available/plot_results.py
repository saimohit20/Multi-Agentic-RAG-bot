"""Thesis-quality figures for the reranker ablation (grownfield).

Reads the per-config judged JSONs (produced by reranker_ablation.py + llm_judge.py)
and renders five publication-style figures. Design follows common academic
conventions:
  * Wong colour-blind-safe palette (Nature Methods, Wong 2011)
  * sans-serif type, coupled font sizes, no top/right spines, light y-grid
  * value labels on bars, border-less legend, white background
  * exported as vector PDF (for LaTeX) + 300 dpi PNG

Figures:
  1. fig1_latency_components   current (v2m3) latency: vector / graph / reranker
  2. fig2_retrieval_quality    MRR, Hit@1, Hit@3 per reranker
  3. fig3_source_identification golden doc found vs not found per reranker
  4. fig4_generation_quality   judge acceptable (correct+partial) + faithfulness
  5. fig5_quality_vs_latency   judge correctness vs total latency (decision chart)

Usage:
  poetry run python -m evaluation.available.plot_results [results_dir]
Default results_dir: evaluation/results/reranker_ablation/grownfield/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- config ----------------------------------------------------------------
CONFIGS = ["no_rerank", "base", "v2m3"]
LABELS = {"no_rerank": "No reranker", "base": "bge-reranker-base", "v2m3": "bge-reranker-v2-m3"}
DEFAULT_RESULTS = Path("evaluation/results/reranker_ablation/grownfield")

# Wong colour-blind-safe palette (Nature Methods, Wong 2011).
WONG = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}
CONFIG_COLOR = {"no_rerank": WONG["blue"], "base": WONG["orange"], "v2m3": WONG["green"]}


def set_style() -> None:
    """Publication rcParams: sans-serif, coupled sizes, minimal chrome."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --- data -------------------------------------------------------------------
def load(results_dir: Path) -> dict[str, list[dict]]:
    data = {}
    for cfg in CONFIGS:
        p = results_dir / f"{cfg}_judged.json"
        if not p.exists():
            raise SystemExit(f"[error] {p} not found — run reranker_ablation + llm_judge first.")
        data[cfg] = json.loads(p.read_text(encoding="utf-8"))
    return data


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(records: list[dict]) -> dict:
    n = len(records) or 1
    ranks = [r.get("golden_rank") for r in records if r.get("golden_rank")]
    verdicts = [r.get("judge_verdict") for r in records]
    return {
        "n": len(records),
        # retrieval
        "mrr": sum(1.0 / r for r in ranks) / n,
        "hit@1": sum(1 for r in ranks if r <= 1) / n,
        "hit@3": sum(1 for r in ranks if r <= 3) / n,
        "found": len(ranks),
        "not_found": len(records) - len(ranks),
        # latency (means, seconds)
        "vector": _mean([r.get("vector_time") for r in records]),
        "graph": _mean([r.get("graph_time") for r in records]),
        "rerank": _mean([r.get("rerank_time") for r in records]),
        "generation": _mean([r.get("generation_time") for r in records]),
        # generation quality
        "n_correct": verdicts.count("correct"),
        "n_partial": verdicts.count("partial"),
        "n_incorrect": verdicts.count("incorrect"),
        "judge_score": _mean([r.get("judge_score") for r in records]),
        "acceptable": (verdicts.count("correct") + verdicts.count("partial")) / n,
        "faithfulness": _mean([r.get("ragas_faithfulness") for r in records]),
    }


# --- figures ----------------------------------------------------------------
def fig1_latency_components(agg: dict, out: Path) -> None:
    """Current (v2m3) latency split into its components — motivates the ablation."""
    a = agg["v2m3"]
    comps = [("Vector", a["vector"], WONG["blue"]),
             ("Graph", a["graph"], WONG["sky"]),
             ("Reranker", a["rerank"], WONG["vermillion"]),
             ("Generation", a["generation"], WONG["green"])]
    names = [c[0] for c in comps]
    vals = [c[1] for c in comps]
    colors = [c[2] for c in comps]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean latency per query (s)")
    ax.set_title("Latency by component (current: bge-reranker-v2-m3)")
    ax.bar_label(bars, fmt="%.1f s", padding=3)
    ax.grid(axis="x", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.margins(x=0.12)
    save(fig, out, "fig1_latency_components")


def fig2_retrieval_quality(agg: dict, out: Path) -> None:
    metrics = [("MRR", "mrr", WONG["blue"]), ("Hit@1", "hit@1", WONG["orange"]),
               ("Hit@3", "hit@3", WONG["green"])]
    x = np.arange(len(CONFIGS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (lab, key, col) in enumerate(metrics):
        vals = [agg[c][key] for c in CONFIGS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Retrieval quality by reranker", pad=34)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    # Legend placed ABOVE the axes so it never overlaps the bars / value labels.
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save(fig, out, "fig2_retrieval_quality")


def fig3_source_identification(agg: dict, out: Path) -> None:
    x = np.arange(len(CONFIGS))
    found = [agg[c]["found"] for c in CONFIGS]
    miss = [agg[c]["not_found"] for c in CONFIGS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x, found, 0.55, label="Golden document found", color=WONG["green"])
    b2 = ax.bar(x, miss, 0.55, bottom=found, label="Not found", color=WONG["vermillion"])
    ax.bar_label(b1, label_type="center", color="white", fontsize=10)
    ax.bar_label(b2, label_type="center", color="white", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_ylabel(f"Questions (of {agg['base']['n']})")
    ax.set_title("Source identification (golden document in top-10)")
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "fig3_source_identification")


def fig4_generation_quality(agg: dict, out: Path) -> None:
    series = [("Acceptable (correct+partial)", "acceptable", WONG["blue"]),
              ("Faithfulness (RAGAS)", "faithfulness", WONG["orange"])]
    x = np.arange(len(CONFIGS))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (lab, key, col) in enumerate(series):
        vals = [agg[c][key] for c in CONFIGS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lab, color=col)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Generation quality by reranker", pad=34)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    # Legend placed ABOVE the axes so it never overlaps the bars / value labels.
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save(fig, out, "fig4_generation_quality")


def fig5_quality_vs_latency(agg: dict, quality: dict, out: Path) -> None:
    """Decision chart: answer quality (mean of RAGAS faithfulness + answer_relevancy)
    vs mean end-to-end latency. `quality` maps config -> score in [0,1]."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for c in CONFIGS:
        q = quality.get(c)
        if q is None:
            continue
        a = agg[c]
        total = a["vector"] + a["graph"] + a["rerank"] + a["generation"]
        ax.scatter(total, q, s=140, color=CONFIG_COLOR[c], zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(LABELS[c], (total, q),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)
    ax.set_xlabel("Latency per query (s)")
    ax.set_ylabel("Answer quality (RAGAS)")
    ax.set_title("Answer quality vs latency")
    ax.grid(color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.margins(0.18)
    # annotate the "better" direction
    ax.text(0.02, 0.02, "← faster", transform=ax.transAxes, color=WONG["grey"], fontsize=9)
    save(fig, out, "fig5_quality_vs_latency")


# --- main -------------------------------------------------------------------
def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    out_dir = results_dir / "figures"
    set_style()
    data = load(results_dir)
    agg = {cfg: aggregate(recs) for cfg, recs in data.items()}

    # Answer-quality for fig5 = mean(faithfulness, answer_relevancy) from the RAGAS
    # pass (reranker_ragas.py). Falls back to the LLM-judge score if RAGAS is absent.
    quality = _load_ragas_quality(results_dir, agg)

    print(f"Rendering figures from {results_dir} -> {out_dir}")
    fig1_latency_components(agg, out_dir)
    fig2_retrieval_quality(agg, out_dir)
    fig3_source_identification(agg, out_dir)
    fig4_generation_quality(agg, out_dir)
    fig5_quality_vs_latency(agg, quality, out_dir)
    print("Done.")


def _load_ragas_quality(results_dir: Path, agg: dict) -> dict:
    """config -> mean(faithfulness, answer_relevancy) from ragas/combined_summary.json.
    Falls back to the judge score (agg[c]['judge_score']) if the RAGAS file is missing."""
    path = results_dir / "ragas" / "combined_summary.json"
    quality: dict = {}
    if path.exists():
        for r in json.loads(path.read_text(encoding="utf-8")):
            f, ar = r.get("faithfulness"), r.get("answer_relevancy")
            if f is not None and ar is not None:
                quality[r["config"]] = (f + ar) / 2
        print(f"  fig5 quality: RAGAS mean(faithfulness, answer_relevancy) from {path}")
    else:
        print(f"  fig5 quality: {path} not found — falling back to judge_score.")
        quality = {c: agg[c].get("judge_score") for c in CONFIGS}
    return quality


if __name__ == "__main__":
    main()

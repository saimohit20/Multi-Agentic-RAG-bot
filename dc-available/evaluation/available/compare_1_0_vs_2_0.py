"""Phase 2 — AVAILABLE 1.0 vs 2.0 comparison plots.

Pure read + plot (no RAG). Reads the finished 1.0 and 2.0 results and renders the
1.0-vs-2.0 figures in canonical thesis style (Wong palette, PNG @300dpi).

Inputs
  1.0 retrieval  new_results/available1.0_results/retrieval_eval/results/<topic>_results.json
  1.0 ragas      new_results/available1.0_results/generation_eval/results/<topic>_ragas_summary.json
  2.0 retrieval  new_results/available2.0_results/<topic>/records.json
  2.0 ragas      new_results/available2.0_results/<topic>/ragas_summary.json

Outputs  ->  new_results/comparison_1.0_vs_2.0/
  retrieval_<topic>.png        A) per-dataset: Hit@1/3/10 + MRR, bars {1.0, 2.0}
  generation_<topic>.png       B) per-dataset: 4 RAGAS metrics, bars {1.0, 2.0}
  summary_retrieval_mrr.png    C) MRR across datasets, bars {1.0, 2.0}
  summary_generation_quality.png  C) mean(faithfulness, answer_relevancy) across datasets
  combined_summary.csv         every (dataset, system) row, all metrics

Usage (from project root):
  poetry run python -m evaluation.available.compare_1_0_vs_2_0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOPICS = [("grownfield", "Grownfield"), ("hydraulics", "Hydraulics"), ("synthetic", "Synthetic")]
K_LIST = [1, 3, 10]
RAGAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RAGAS_LABELS = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]

# Wong palette: 1.0 = light blue (baseline), 2.0 = orange (improved).
WONG = {"grey": "#999999", "green": "#009E73", "blue": "#0072B2",
        "orange": "#E69F00", "sky": "#56B4E9"}
SYS_COLOR = {"1.0": WONG["sky"], "2.0": WONG["orange"]}
SYS_ORDER = ["1.0", "2.0"]

BASE = Path("evaluation/new_results")
DIR_1 = BASE / "available1.0_results"
DIR_2 = BASE / "available2.0_results"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


# --- loading -----------------------------------------------------------------
def _ranks(path: Path) -> list:
    recs = json.loads(path.read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    return [r.get("golden_rank") for r in recs]


def _retrieval_metrics(ranks: list) -> dict:
    n = len(ranks) or 1
    m = {f"hit@{k}": sum(1 for rk in ranks if rk and rk <= k) / n for k in K_LIST}
    m["mrr"] = sum(1.0 / rk for rk in ranks if rk) / n
    return m


def _ragas(path: Path, topic: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d = d.get(topic, d)          # 1.0 is nested {topic:{...}}, 2.0 is flat
    return {m: float(d.get(m)) for m in RAGAS if d.get(m) is not None}


def load_all() -> dict:
    """topic -> {'retrieval': {sys: metrics}, 'ragas': {sys: metrics}}."""
    out = {}
    for topic, _ in TOPICS:
        r1 = _retrieval_metrics(_ranks(DIR_1 / "retrieval_eval" / "results" / f"{topic}_results.json"))
        r2 = _retrieval_metrics(_ranks(DIR_2 / topic / "records.json"))
        g1 = _ragas(DIR_1 / "generation_eval" / "results" / f"{topic}_ragas_summary.json", topic)
        g2 = _ragas(DIR_2 / topic / "ragas_summary.json", topic)
        out[topic] = {"retrieval": {"1.0": r1, "2.0": r2}, "ragas": {"1.0": g1, "2.0": g2}}
    return out


# --- generic grouped-bar helper ---------------------------------------------
def grouped_bars(xlabels, series, out: Path, title, ylabel="Score (0–1)", ylim=(0, 1.05)) -> None:
    """series = [(sys_name, [values...]), ...] in SYS_ORDER."""
    x = np.arange(len(xlabels))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(xlabels) + 2), 4.4))
    for i, (name, vals) in enumerate(series):
        offs = x + (i - (len(series) - 1) / 2) * w
        bars = ax.bar(offs, vals, w, label=f"AVAILABLE {name}", color=SYS_COLOR[name])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=30)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, ncol=len(series), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(out)
    plt.close(fig)


# --- figures -----------------------------------------------------------------
def plot_per_dataset(data: dict, out_dir: Path) -> None:
    rkeys = [f"hit@{k}" for k in K_LIST] + ["mrr"]
    rlabels = [f"Hit@{k}" for k in K_LIST] + ["MRR"]
    for topic, label in TOPICS:
        # A) retrieval
        grouped_bars(
            rlabels,
            [(s, [data[topic]["retrieval"][s][k] for k in rkeys]) for s in SYS_ORDER],
            out_dir / f"retrieval_{topic}.png",
            f"Retrieval — {label}: AVAILABLE 1.0 vs 2.0",
        )
        # B) generation (RAGAS 4)
        grouped_bars(
            RAGAS_LABELS,
            [(s, [data[topic]["ragas"][s].get(m, 0.0) for m in RAGAS]) for s in SYS_ORDER],
            out_dir / f"generation_{topic}.png",
            f"Generation (RAGAS) — {label}: AVAILABLE 1.0 vs 2.0",
        )


def plot_headlines(data: dict, out_dir: Path) -> None:
    labels = [lbl for _, lbl in TOPICS]
    # C1) MRR across datasets
    grouped_bars(
        labels,
        [(s, [data[t]["retrieval"][s]["mrr"] for t, _ in TOPICS]) for s in SYS_ORDER],
        out_dir / "summary_retrieval_mrr.png",
        "Retrieval MRR — AVAILABLE 1.0 vs 2.0",
        ylabel="MRR",
    )
    # C2) answer quality = mean(faithfulness, answer_relevancy) across datasets
    def q(topic, s):
        g = data[topic]["ragas"][s]
        return (g.get("faithfulness", 0.0) + g.get("answer_relevancy", 0.0)) / 2
    grouped_bars(
        labels,
        [(s, [q(t, s) for t, _ in TOPICS]) for s in SYS_ORDER],
        out_dir / "summary_generation_quality.png",
        "Answer quality (RAGAS) — AVAILABLE 1.0 vs 2.0",
        ylabel="mean(faithfulness, answer_relevancy)",
    )


def load_latency_quality() -> dict:
    """topic -> {'latency': {sys: s}, 'quality': {sys: 0-1}}.
    1.0 from timing.json (its own run); 2.0 from records.json + ragas_summary."""
    timing = json.loads((DIR_1 / "timing.json").read_text(encoding="utf-8"))
    out = {}
    for topic, _ in TOPICS:
        t1 = timing[topic]
        recs = json.loads((DIR_2 / topic / "records.json").read_text(encoding="utf-8"))
        tot2 = [r.get("retrieval_time", 0.0) + r.get("generation_time", 0.0) for r in recs]
        lat2 = sum(tot2) / (len(tot2) or 1)
        g2 = _ragas(DIR_2 / topic / "ragas_summary.json", topic)
        q2 = (g2.get("faithfulness", 0.0) + g2.get("answer_relevancy", 0.0)) / 2
        out[topic] = {
            "latency": {"1.0": t1["total_time_mean"], "2.0": lat2},
            "quality": {"1.0": t1["answer_quality"], "2.0": q2},
        }
    return out


def plot_latency(lq: dict, out: Path) -> None:
    labels = [lbl for _, lbl in TOPICS]
    hi = max(lq[t]["latency"][s] for t, _ in TOPICS for s in SYS_ORDER)
    grouped_bars(
        labels,
        [(s, [lq[t]["latency"][s] for t, _ in TOPICS]) for s in SYS_ORDER],
        out,
        "Mean latency per query — AVAILABLE 1.0 vs 2.0",
        ylabel="Latency per query (s)",
        ylim=(0, hi * 1.18),
    )


def plot_quality_vs_latency(lq: dict, out: Path) -> None:
    """Scatter: x=latency, y=answer quality. 1.0 = circle, 2.0 = square, per dataset,
    joined by a line to show the trade-off shift."""
    from matplotlib.lines import Line2D
    dcolor = {"grownfield": WONG["blue"], "hydraulics": WONG["green"], "synthetic": WONG["orange"]}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for topic, label in TOPICS:
        x1, y1 = lq[topic]["latency"]["1.0"], lq[topic]["quality"]["1.0"]
        x2, y2 = lq[topic]["latency"]["2.0"], lq[topic]["quality"]["2.0"]
        c = dcolor[topic]
        ax.plot([x1, x2], [y1, y2], color=c, alpha=0.45, lw=1.3, zorder=1)
        ax.scatter([x1], [y1], marker="o", s=130, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([x2], [y2], marker="s", s=130, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(label, ((x1 + x2) / 2, (y1 + y2) / 2),
                    textcoords="offset points", xytext=(6, 6), fontsize=9, color=c)
    ax.set_xlabel("Latency per query (s)")
    ax.set_ylabel("Answer quality (RAGAS)")
    ax.set_title("Answer quality vs latency — AVAILABLE 1.0 vs 2.0")
    ax.grid(color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.margins(0.15)
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=WONG["grey"], markersize=10, label="1.0"),
              Line2D([0], [0], marker="s", color="w", markerfacecolor=WONG["grey"], markersize=10, label="2.0")]
    ax.legend(handles=legend, frameon=False, loc="lower right", title="System")
    fig.savefig(out)
    plt.close(fig)


def plot_latency_quality(lq: dict, out: Path) -> None:
    """One plot: latency = bars (left axis), answer quality = diamonds (right axis),
    for 1.0 vs 2.0 across datasets. Each diamond sits above its system's bar."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    labels = [lbl for _, lbl in TOPICS]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    hi = max(lq[t]["latency"][s] for t, _ in TOPICS for s in SYS_ORDER)

    # latency bars (left axis)
    pos = {}
    for i, s in enumerate(SYS_ORDER):
        xp = x + (i - 0.5) * w
        pos[s] = xp
        bars = ax.bar(xp, [lq[t]["latency"][s] for t, _ in TOPICS], w,
                      color=SYS_COLOR[s], zorder=2)
        ax.bar_label(bars, fmt="%.1fs", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency per query (s)")
    ax.set_ylim(0, hi * 1.22)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)

    # answer quality diamonds (right axis), above each system's bar
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Answer quality (RAGAS)")
    ax2.spines["top"].set_visible(False)
    for s in SYS_ORDER:
        q = [lq[t]["quality"][s] for t, _ in TOPICS]
        ax2.plot(pos[s], q, marker="D", ms=10, ls="none",
                 color="#333333", markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        for xi, qi in zip(pos[s], q):
            ax2.annotate(f"{qi:.2f}", (xi, qi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color="#333333")

    ax.set_title("Latency and answer quality — AVAILABLE 1.0 vs 2.0", pad=30)
    handles = [
        Patch(color=SYS_COLOR["1.0"], label="1.0 latency"),
        Patch(color=SYS_COLOR["2.0"], label="2.0 latency"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#333333",
               markersize=9, label="Answer quality (right axis)"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(out)
    plt.close(fig)


def write_csv(data: dict, out: Path) -> None:
    cols = [f"hit@{k}" for k in K_LIST] + ["mrr"] + RAGAS
    lines = ["dataset,system," + ",".join(cols)]
    for topic, _ in TOPICS:
        for s in SYS_ORDER:
            r = data[topic]["retrieval"][s]
            g = data[topic]["ragas"][s]
            vals = [r[f"hit@{k}"] for k in K_LIST] + [r["mrr"]] + [g.get(m, float("nan")) for m in RAGAS]
            lines.append(f"{topic},{s}," + ",".join(f"{v:.4f}" for v in vals))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "comparison_1.0_vs_2.0"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    data = load_all()
    plot_per_dataset(data, out_dir)
    plot_headlines(data, out_dir)
    write_csv(data, out_dir / "combined_summary.csv")

    # latency + quality-vs-latency (1.0 timing.json vs 2.0 records.json)
    lq = load_latency_quality()
    plot_latency(lq, out_dir / "summary_latency.png")
    plot_quality_vs_latency(lq, out_dir / "quality_vs_latency.png")
    plot_latency_quality(lq, out_dir / "latency_and_quality.png")
    print(f"Wrote 6 per-dataset + 2 headline + 3 latency/quality plots + combined_summary.csv to {out_dir}")


if __name__ == "__main__":
    main()

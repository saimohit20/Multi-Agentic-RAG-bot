"""Section 1b retrieval figures and metric tables.

Produces the vector/graph/hybrid comparison charts in the reference grouped-bar
style (grey / orange / blue, horizontal gridlines, value labels, no title) plus
the underlying metric tables.

Inputs (read only, nothing is overwritten):
  evaluation/results/synthetic/{vector,graph,hybrid}.json      per-leg records
  evaluation/results/sample results/final_retrieval_2026-08-06_22-41/metrics.json
  evaluation/results/available_2.0/hydraulics/metrics.json

Outputs (evaluation/results/plots_1b/):
  grownfield_metrics.png        Hit@1/3/10 + MRR, three modes (code level)
  hydraulics_metrics.png        Hit@1/3/10 + MRR, three modes
  synthetic_metrics.png         Hit@1/3/10 + MRR, three modes (overall)
  synthetic_hit10_by_type.png   Hit@10 by the five synthetic question types
  metrics_1b.json / metrics_1b.csv   all values, full precision

Synthetic scoring is recomputed from the per-leg files with the same name
matcher as rescore.py, so the overall figures reproduce retrieval_metrics.json
exactly. The stale golden_rank in retrieval.json is not used.

Run from the project root (dc-available):
  poetry run python -m evaluation.available.common.plots_1b
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("evaluation/results")
SYN = RES / "synthetic"
GF_METRICS = RES / "sample results" / "final_retrieval_2026-08-06_22-41" / "metrics.json"
HYD_METRICS = RES / "available_2.0" / "hydraulics" / "metrics.json"
OUT = RES / "plots_1b"

LEGS = ["vector", "graph", "hybrid"]
LEG_LABEL = {"vector": "Vector", "graph": "Graph", "hybrid": "Hybrid"}
LEG_COLOR = {"vector": "#A5A5A5", "graph": "#ED7D31", "hybrid": "#4472C4"}

METRIC_KEYS = ["hit@1", "hit@3", "hit@10", "mrr"]
METRIC_LABEL = ["Hit@1", "Hit@3", "Hit@10", "MRR"]
SYN_TYPES = ["factual", "conceptual", "procedural", "application", "summary"]

GRID = "#D9D9D9"

# --- synthetic scoring (same matcher as rescore.py) -------------------------
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def _norm(value) -> str:
    s = _EXT_RE.sub("", str(value or "").strip()).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _rank_of(golden, docs):
    key = _norm(golden)
    if not key:
        return None
    for i, name in enumerate(docs, 1):
        if _norm(name) == key:
            return i
    return None


def _agg(rows) -> dict:
    n = len(rows) or 1
    ranks = [_rank_of(r.get("golden_source", ""), r.get("retrieved_docs", [])) for r in rows]
    out = {f"hit@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 4)
           for k in (1, 3, 10)}
    out["mrr"] = round(sum(1.0 / x for x in ranks if x) / n, 4)
    out["n"] = len(rows)
    return out


def load_synthetic():
    legs = {m: json.loads((SYN / f"{m}.json").read_text(encoding="utf-8")) for m in LEGS}
    overall = {m: _agg(legs[m]) for m in LEGS}
    by_type = {t: {m: _agg([r for r in legs[m] if r.get("question_type") == t]) for m in LEGS}
               for t in SYN_TYPES}
    return overall, by_type


def load_grownfield():
    o = json.loads(GF_METRICS.read_text(encoding="utf-8"))["overall"]
    return {m: {"hit@1": o[m]["hit@1_code"], "hit@3": o[m]["hit@3_code"],
                "hit@10": o[m]["hit@10_code"], "mrr": o[m]["mrr_code"]} for m in LEGS}


def load_hydraulics():
    o = json.loads(HYD_METRICS.read_text(encoding="utf-8"))
    return {m: {k: o[m][k] for k in METRIC_KEYS} for m in LEGS}


# --- plotting ---------------------------------------------------------------
def _style(ax):
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0)


def _grouped_bar(cats, series, xlabels, fname):
    """series: {mode: [value per category]}."""
    x = np.arange(len(cats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(cats)), 4.6))
    for i, m in enumerate(LEGS):
        offs = x + (i - 1) * w
        bars = ax.bar(offs, series[m], w, label=LEG_LABEL[m], color=LEG_COLOR[m])
        for b, v in zip(bars, series[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    _style(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_metric_panel(metrics, fname):
    series = {m: [metrics[m][k] for k in METRIC_KEYS] for m in LEGS}
    _grouped_bar(METRIC_KEYS, series, METRIC_LABEL, fname)


def fig_synthetic_by_type(by_type, fname):
    series = {m: [by_type[t][m]["hit@10"] for t in SYN_TYPES] for m in LEGS}
    _grouped_bar(SYN_TYPES, series, [t.capitalize() for t in SYN_TYPES], fname)


# --- tables -----------------------------------------------------------------
def write_tables(syn_overall, syn_by_type, gf, hyd):
    payload = {
        "grownfield_code": gf,
        "hydraulics": hyd,
        "synthetic_overall": syn_overall,
        "synthetic_by_type": syn_by_type,
    }
    (OUT / "metrics_1b.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (OUT / "metrics_1b.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "split", "mode", "hit@1", "hit@3", "hit@10", "mrr", "n"])

        def row(dataset, split, mode, d):
            wr.writerow([dataset, split, mode, d.get("hit@1", ""), d.get("hit@3", ""),
                         d.get("hit@10", ""), d.get("mrr", ""), d.get("n", "")])

        for m in LEGS:
            row("grownfield_code", "overall", m, gf[m])
        for m in LEGS:
            row("hydraulics", "overall", m, hyd[m])
        for m in LEGS:
            row("synthetic", "overall", m, syn_overall[m])
        for t in SYN_TYPES:
            for m in LEGS:
                row("synthetic", t, m, syn_by_type[t][m])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syn_overall, syn_by_type = load_synthetic()
    gf = load_grownfield()
    hyd = load_hydraulics()

    fig_metric_panel(gf, "grownfield_metrics.png")
    fig_metric_panel(hyd, "hydraulics_metrics.png")
    fig_metric_panel(syn_overall, "synthetic_metrics.png")
    fig_synthetic_by_type(syn_by_type, "synthetic_hit10_by_type.png")
    write_tables(syn_overall, syn_by_type, gf, hyd)

    print("synthetic overall vector:", syn_overall["vector"])
    print("grownfield code vector  :", gf["vector"])
    print("hydraulics vector       :", hyd["vector"])
    print(f"Wrote 4 figures + metrics_1b.json/csv to {OUT}")


if __name__ == "__main__":
    main()

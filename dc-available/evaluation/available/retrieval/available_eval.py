"""Available 2.0 evaluation: compare the three retrieval legs on a topic.

Legs (each is a data_config override set, injected by the topic scripts):
    vector   FUSION_LEGS=vector                       — dense passages only
    graph    FUSION_LEGS=graph  (PPR, merge seeds)    — graph leg only
    hybrid   FUSION_LEGS=both   (vector + PPR-merge)  — the Available 2.0 path

For every question x leg we record, at DOCUMENT level:
    golden_rank  position of the golden document AFTER the reranker
                 (unique docs of the top-FUSION_OUTPUT_K reranked chunks)

Output goes to  results/available_2.0/<topic>/ :
    <topic>.json                one record per (question x retrieval_type), long format
    metrics.json / metrics.csv  Hit@k + MRR per leg (post-rerank, document level)
    plot_identification.png     golden identified in top-10 per leg
    plot_retrieval_metrics.png  Hit@10 + MRR per leg
    (plot_generation_ragas.png  later — needs generated answers)

generated_* fields are written as null here; the RAGAS/generation pass fills them
later. `retrieved_contexts` (chunk texts) is captured now so RAGAS can reuse it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.available.retrieval import common as gc   # golden matching, DatasetSpec, constants

# --- canonical thesis plot style (see EVALUATION_PLAN.md / plot_results.py) ---
# Wong colour-blind-safe palette (Nature Methods, Wong 2011). PNG @300dpi only.
WONG = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}


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


# --- dataset loading (richer than seeding/engine: carries answer + module) ----
def load_rows(spec: gc.DatasetSpec) -> list[dict]:
    path = spec.dataset_path
    df = pd.read_excel(path) if str(path).lower().endswith(".xlsx") else pd.read_csv(path)

    def col(name):
        return name if name and name in df.columns else None

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        qid = str(r.get(spec.col_qid)) if col(spec.col_qid) else f"q{i + 1}"
        rows.append({
            "question_id": qid,
            "question": str(r.get(spec.col_question)),
            "question_type": str(r.get(spec.col_qtype, "")).strip().lower() if col(spec.col_qtype) else "",
            "level": str(r.get("level", "")).strip() if "level" in df.columns else "",
            "type": str(r.get("type", "")).strip() if "type" in df.columns else "",
            "golden_source": sorted(gc.golden_of(r.get(spec.col_golden), spec.match_mode)),
            "golden_answer": str(r.get(spec.col_answer)) if col(spec.col_answer) else "",
            "module": str(r.get(spec.col_module, "")).strip() if col(spec.col_module) else "",
            "submodule": str(r.get(spec.col_submodule, "")).strip() if col(spec.col_submodule) else "",
            "_golden": gc.golden_of(r.get(spec.col_golden), spec.match_mode),  # set, for matching
        })
    return rows


# --- ranking helper ----------------------------------------------------------
def _rank_of(golden: set[str], docs: list[str], match_mode: str) -> int | None:
    for i, name in enumerate(docs, 1):
        if gc._doc_keys(name, match_mode) & golden:
            return i
    return None


def _unique_docs(chunks, PART_OF_DOCUMENT, k: int) -> list[str]:
    seen, out = set(), []
    for d in chunks:
        doc = d.metadata.get(PART_OF_DOCUMENT) or "N/A"
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
        if len(out) >= k:
            break
    return out


# --- retrieval (slow; needs the RAG stack) -----------------------------------
def run_retrieval(rows, legs, spec) -> list[dict]:
    """Return long-format records: one per (question, leg)."""
    here = Path(__file__).resolve().parent          # .../src/graph
    bad = {str(here), str(here.parents[1])}          # .../src/graph, .../evaluation
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in bad]

    from dc_available.config import data_config
    data_config.update_data_config()
    data_config.RETRIEVAL_MODE = "fusion"
    from dc_available.RAG import qa_functions as qa
    from dc_available.RAG.data import AvailableMetaData

    PART = AvailableMetaData.PART_OF_DOCUMENT
    is_grownfield = bool(spec.col_module)   # only grownfield carries module/submodule
    records = []

    for leg, overrides in legs.items():
        print(f"\n=== leg: {leg} ({overrides}) ===")
        for row in rows:
            q = row["question"]
            golden = row["_golden"]
            # retrieval — the real (post-rerank) answer path, one pass per question
            for key, val in overrides.items():
                setattr(data_config, key, val)
            data_config.RERANKER_ENABLED = True
            data_config.FUSION_OUTPUT_K = gc.TOP_DOCS
            try:
                final = qa.search_question(q)
            except Exception as exc:
                print(f"  {row['question_id']}/{leg}: ERROR {exc}")
                final = []
            post_docs = _unique_docs(final, PART, gc.TOP_DOCS)
            contexts = [(d.page_content or "") for d in final]

            rec = {
                "question_id": row["question_id"],
                "question": q,
                "question_type": row["question_type"],
                "level": row["level"],
                "type": row["type"],
                "golden_source": row["golden_source"],
                "golden_answer": row["golden_answer"],
                "retrieval_type": leg,
                "retrieved_docs": post_docs,
                "golden_rank": _rank_of(golden, post_docs, spec.match_mode),
                "retrieved_contexts": contexts,
                "generated_answer": None,      # RAGAS/generation pass fills this later
            }
            if is_grownfield:
                rec.update({
                    "module": row["module"], "submodule": row["submodule"],
                    "generated_module": None, "generated_submodule": None,
                })
            records.append(rec)
            print(f"  {row['question_id']}/{leg}: rank={rec['golden_rank']} ({len(post_docs)} docs)")

    return records


# --- metrics (post-rerank, document level) -----------------------------------
def compute_metrics(records, legs) -> dict:
    metrics = {}
    for leg in legs:
        recs = [r for r in records if r["retrieval_type"] == leg]
        n = len(recs) or 1
        m = {}
        for k in gc.K_LIST:
            m[f"hit@{k}"] = round(sum(1 for r in recs
                                      if r["golden_rank"] and r["golden_rank"] <= k) / n, 4)
        m["mrr"] = round(sum(1.0 / r["golden_rank"] for r in recs if r["golden_rank"]) / n, 4)
        metrics[leg] = m
    return metrics


# --- plots -------------------------------------------------------------------
def plot_identification(records, legs, labels, out: Path) -> None:
    legs = list(legs)
    n = len({r["question_id"] for r in records}) or 1
    found = [int(sum(1 for r in records
                     if r["retrieval_type"] == leg and r["golden_rank"] and r["golden_rank"] <= gc.TOP_DOCS))
             for leg in legs]
    miss = [n - x for x in found]
    x = np.arange(len(legs))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x, found, 0.55, label="Golden document found", color=WONG["green"])
    b2 = ax.bar(x, miss, 0.55, bottom=found, label="Not found", color=WONG["vermillion"])
    ax.bar_label(b1, label_type="center", color="white", fontsize=10)
    ax.bar_label(b2, label_type="center", color="white", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[l] for l in legs])
    ax.set_ylabel(f"Questions (of {n})")
    ax.set_title(f"Source identification (golden document in top-{gc.TOP_DOCS})")
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "plot_identification.png")
    plt.close(fig)


def plot_retrieval_metrics(metrics, legs, labels, out: Path) -> None:
    legs = list(legs)
    bars = [("hit@10", "Hit@10", WONG["blue"]), ("mrr", "MRR", WONG["orange"])]
    x = np.arange(len(legs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (key, lab, color) in enumerate(bars):
        offs = x + (i - 0.5) * w
        vals = [metrics[l][key] for l in legs]
        rects = ax.bar(offs, vals, w, label=lab, color=color)
        ax.bar_label(rects, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[l] for l in legs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Retrieval quality by leg — Hit@10 and MRR", pad=34)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(out / "plot_retrieval_metrics.png")
    plt.close(fig)


# --- orchestrator ------------------------------------------------------------
def evaluate(spec: gc.DatasetSpec, legs, labels, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(spec)
    topic_json = out_dir / f"{spec.name}.json"

    if topic_json.exists():
        print(f"Reusing {topic_json} (skipping retrieval).")
        records = json.loads(topic_json.read_text(encoding="utf-8"))
    else:
        records = run_retrieval(rows, legs, spec)
        topic_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = compute_metrics(records, legs)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).T.to_csv(out_dir / "metrics.csv")

    print("\nleg      Hit@1 Hit@3 Hit@10  MRR")
    for leg in legs:
        m = metrics[leg]
        print(f"{leg:8}  {m['hit@1']:.2f}  {m['hit@3']:.2f}   {m['hit@10']:.2f}  {m['mrr']:.2f}")

    set_style()
    plot_identification(records, legs, labels, out_dir)
    plot_retrieval_metrics(metrics, legs, labels, out_dir)

    print(f"\nWrote to {out_dir}:")
    for f in (f"{spec.name}.json", "metrics.json", "metrics.csv",
              "plot_identification.png", "plot_retrieval_metrics.png"):
        print(f"  {f}")
    return metrics


def cli_main(spec: gc.DatasetSpec, legs, labels) -> None:
    """  python <topic>/available.py [dataset_path] [out_dir]

    Default out_dir = results/available_2.0/<topic>/ (fixed folder, per your layout)."""
    if len(sys.argv) > 1:
        spec = replace(spec, dataset_path=Path(sys.argv[1]))
    out = (Path(sys.argv[2]) if len(sys.argv) > 2
           else Path("evaluation/results/available_2.0") / spec.name)
    evaluate(spec, legs, labels, out)


# --- combined cross-topic plot (hybrid leg) ----------------------------------
# Folded in from the former graph_combined.py: reads the per-topic <topic>.json
# files and draws two grouped-bar charts with the TOPICS on the x-axis. Pure
# read + plot, so run it after the per-topic evals exist.
_COMBINED_LEG = "hybrid"
_COMBINED_IDENT_K = gc.TOP_DOCS      # golden doc in top-10 (chunks the LLM reads)
_COMBINED_HIT_K = 5                  # Hit@5 for the ranking-quality plot
_COMBINED_TOPICS = [("grownfield", "Grownfield"), ("hydraulics", "Hydraulics")]
_BLUE, _ORANGE = WONG["blue"], WONG["orange"]


def _combined_stats(records: list[dict]) -> dict:
    n = len(records)
    ranks = [r.get("golden_rank") for r in records]
    identified = sum(1 for rk in ranks if rk and rk <= _COMBINED_IDENT_K)
    hit_k = sum(1 for rk in ranks if rk and rk <= _COMBINED_HIT_K) / (n or 1)
    mrr = sum(1.0 / rk for rk in ranks if rk) / (n or 1)
    return {"n": n, "identified": identified, "not_identified": n - identified,
            "hit_k": round(hit_k, 4), "mrr": round(mrr, 4)}


def _combined_grouped(labels, series, out: Path, title, ylabel, ylim=None, fmt="%.0f") -> None:
    """series = [(name, color, values-per-topic), ...]."""
    set_style()
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (name, color, vals) in enumerate(series):
        offs = x + (i - (len(series) - 1) / 2) * w
        rects = ax.bar(offs, vals, w, label=name, color=color)
        ax.bar_label(rects, fmt=fmt, padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, pad=34)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, ncol=len(series), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(out)
    plt.close(fig)


def plot_combined(results_dir: Path = Path("evaluation/results/available_2.0"),
                  topics=_COMBINED_TOPICS) -> None:
    """Cross-topic combined plots (hybrid leg) from the per-topic <topic>.json files.

    Run after the per-topic evals (grownfield/hydraulics) have produced their JSONs.
    Skips (with a message) if a topic's JSON is missing, so a partial run is safe.
    """
    labels, stats = [], []
    for topic, label in topics:
        path = results_dir / topic / f"{topic}.json"
        if not path.exists():
            print(f"[plot_combined] missing {path} — run the {topic} eval first; skipping.")
            return
        records = json.loads(path.read_text(encoding="utf-8"))
        hybrid = [r for r in records if r.get("retrieval_type") == _COMBINED_LEG]
        if not hybrid:
            print(f"[plot_combined] no '{_COMBINED_LEG}' records in {path}; skipping.")
            return
        labels.append(label)
        stats.append(_combined_stats(hybrid))

    _combined_grouped(
        labels,
        [("Identified", _BLUE, [s["identified"] for s in stats]),
         ("Not identified", _ORANGE, [s["not_identified"] for s in stats])],
        results_dir / "plot_source_identification.png",
        title=f"Source identification (golden doc in top-{_COMBINED_IDENT_K}) — {_COMBINED_LEG}",
        ylabel="No. of questions",
    )
    _combined_grouped(
        labels,
        [(f"Hit@{_COMBINED_HIT_K}", _BLUE, [s["hit_k"] for s in stats]),
         ("MRR", _ORANGE, [s["mrr"] for s in stats])],
        results_dir / "plot_retrieval_metrics.png",
        title=f"Retrieval metrics — {_COMBINED_LEG}",
        ylabel="Score", ylim=(0, 1.05), fmt="%.2f",
    )
    print(f"[plot_combined] wrote plot_source_identification.png + "
          f"plot_retrieval_metrics.png to {results_dir}")

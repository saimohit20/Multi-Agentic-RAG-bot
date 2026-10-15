"""Graph-leg evaluation: 2 engines x 2 seedings (graph-only, no vector leak).

Retrieval-only (no answer generation). For each config x question it records the
top-10 retrieved DOCUMENTS, builds the comparison dataframe, computes Hit@k + MRR,
and writes two plots:

  * plot_identified.png — per config, how many questions had the golden source
    identified (in top-10) vs not.
  * plot_metrics.png    — Hit@1/3/5/10 and MRR per config.

All four isolate the graph engine (FUSION_LEGS="graph", no vector seeds leak in).
The 2x2 grid answers two questions in one run:
  entity_1hop_legacy -> 1-hop engine, old top-8-by-fused seeds
  entity_1hop_merge  -> 1-hop engine, new threshold+merge seeds
  ppr_legacy         -> PPR engine,   old top-8-by-fused seeds
  ppr_merge          -> PPR engine,   new threshold+merge seeds
  * seeding effect: legacy -> merge (within each engine)
  * engine effect : entity_1hop -> ppr (within the merge column, identical seeds)

Usage:
  poetry run python -m evaluation.available.retrieval.graph_ablation [topic] [out_dir]
    topic   "grownfield" (default) | "hydraulics"
  Default out_dir: evaluation/new_results/graph_ablation/<topic>/
"""
from __future__ import annotations

import os
import sys

# Running "python evaluation/…" puts evaluation/ on sys.path[0], which lets the
# local "datasets" folder shadow HuggingFace's "datasets" (imported by
# FlagEmbedding). Drop the script dir before any dc_available import.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import json
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- canonical thesis plot style (shared with plot_results.py) --------------
# Wong colour-blind-safe palette (Nature Methods, Wong 2011).
WONG = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}
LABELS = {"entity_1hop": "Entity 1-hop", "entity_3hop": "Entity 3-hop", "ppr": "PPR"}


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

# NOTE: dc_available (qa_functions / data_config) is imported lazily inside the
# retrieval branch of main() — importing qa_functions runs init_global_vars(),
# which boots the whole RAG stack (Qdrant, embeddings). Keeping it lazy lets the
# cache-reuse path (recompute metrics + redraw plots from comparison.json) run
# with NO RAG stack / network.
from evaluation.available.retrieval import common as gc   # golden matching helpers

K_LIST = [1, 5, 10]
TOP_DOCS = 10

# Topic presets: dataset + how golden is matched.
#   "code"     Grownfield source codes (O2C-02.010)
#   "filename" a single source file     (hydraulics)
TOPICS: dict[str, dict] = {
    "grownfield": {
        "dataset": "evaluation/datasets/Grownfield/dataset_full.xlsx",
        "match_mode": "code", "golden_col": "golden_codes",
    },
    "hydraulics": {
        "dataset": "evaluation/datasets/hydraluics/01_Teilnehmerunterlage_RT04-004_DE_dataset.xlsx",
        "match_mode": "filename", "golden_col": "source",
    },
}
MATCH_MODE = "code"   # overwritten from the chosen topic in main()

# config name -> data_config overrides (applied at runtime; read per call).
# Every config sets all four discriminating knobs so runs can't leak into each
# other regardless of order. Graph-only (FUSION_LEGS="graph"); module isolation
# and topic isolation are governed globally (GRAPH_HIERARCHICAL_ENABLED=False,
# PPR_TOPIC_ISOLATION=True) and held constant across all four.
# Three graph engines, all with merge seeding (legacy seeding dropped). The entity
# engine's walk depth is set by GRAPH_ENTITY_HOP_K (1 vs 3); PPR is its own engine.
CONFIGS: dict[str, dict] = {
    "entity_1hop": {"FUSION_LEGS": "graph", "GRAPH_RETRIEVAL_MODE": "entity_1hop",
                    "GRAPH_ENTITY_HOP_K": 1,
                    "GRAPH_ENTITY_HOP_SEED_MODE": "merge", "PPR_SEED_MODE": "merge"},
    "entity_3hop": {"FUSION_LEGS": "graph", "GRAPH_RETRIEVAL_MODE": "entity_1hop",
                    "GRAPH_ENTITY_HOP_K": 3,
                    "GRAPH_ENTITY_HOP_SEED_MODE": "merge", "PPR_SEED_MODE": "merge"},
    "ppr":         {"FUSION_LEGS": "graph", "GRAPH_RETRIEVAL_MODE": "ppr",
                    "GRAPH_ENTITY_HOP_SEED_MODE": "merge", "PPR_SEED_MODE": "merge"},
}
COLORS = {
    "entity_1hop": "#e69f00",
    "entity_3hop": "#0072b2",
    "ppr":         "#009e73",
}


def _golden(value) -> set[str]:
    """Golden document keys for a dataset row (codes or filename stem)."""
    return gc.golden_of(value, MATCH_MODE)


def _ranked_documents(docs, part: str, top: int = TOP_DOCS) -> list[str]:
    """Unique document names from the retrieved chunks, in rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        name = d.metadata.get(part) or "N/A"
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= top:
            break
    return out


def _hit_at_k(golden: set[str], retrieved: list[str], k: int) -> float:
    return 1.0 if any(gc._doc_keys(n, MATCH_MODE) & golden for n in retrieved[:k]) else 0.0


def _mrr(golden: set[str], retrieved: list[str]) -> float:
    for rank, name in enumerate(retrieved, 1):
        if gc._doc_keys(name, MATCH_MODE) & golden:
            return 1.0 / rank
    return 0.0


def _apply(data_config, overrides: dict) -> None:
    for key, value in overrides.items():
        setattr(data_config, key, value)


def main() -> None:
    global MATCH_MODE

    # argv[1] = topic ("grownfield" | "hydraulics"); argv[2] = out dir (optional).
    topic = sys.argv[1] if len(sys.argv) > 1 else "grownfield"
    if topic not in TOPICS:
        raise SystemExit(f"unknown topic '{topic}' — choose from {list(TOPICS)}")
    preset = TOPICS[topic]
    MATCH_MODE = preset["match_mode"]
    inp = Path(preset["dataset"])
    gcol = preset["golden_col"]
    out = (Path(sys.argv[2]) if len(sys.argv) > 2
           else Path("evaluation/new_results/graph_ablation") / topic)
    out.mkdir(parents=True, exist_ok=True)

    print(f"topic={topic}  match_mode={MATCH_MODE}  golden_col={gcol}\ndataset={inp}\nout={out}")
    df_in = pd.read_excel(inp) if str(inp).lower().endswith(".xlsx") else pd.read_csv(inp)
    if gcol not in df_in.columns:
        raise SystemExit(f"golden column '{gcol}' not in {inp} (have {list(df_in.columns)})")
    # question_id may be absent (hydraulics) -> fall back to a row-index id so each
    # question stays a distinct key in the retrieved dict.
    has_qid = "question_id" in df_in.columns
    questions = [
        (str(r["question_id"]) if has_qid and pd.notna(r.get("question_id")) else f"q{i + 1}",
         str(r.get("question")), r.get(gcol))
        for i, (_, r) in enumerate(df_in.iterrows())
    ]

    comp_path = out / "comparison.json"
    if comp_path.exists():
        # --- reuse cached retrieval (no RAG stack) -------------------------
        print(f"Reusing {comp_path} (skipping retrieval; recomputing metrics + plots).")
        comparison_json = json.loads(comp_path.read_text(encoding="utf-8"))
        qids = [e["question_id"] for e in comparison_json]
        golden_by_qid = {e["question_id"]: set(e["golden_source"]) for e in comparison_json}
        retrieved = {c: {e["question_id"]: e.get(c, []) for e in comparison_json} for c in CONFIGS}
    else:
        # --- run retrieval for each config (boots the RAG stack) -----------
        # Lazy imports: importing qa_functions runs init_global_vars() (Qdrant +
        # embeddings), so we only pay that when there is no cache to reuse.
        from dc_available.config import data_config
        data_config.update_data_config()
        data_config.RETRIEVAL_MODE = "fusion"
        from dc_available.RAG import qa_functions as qa
        from dc_available.RAG.data import AvailableMetaData
        part = AvailableMetaData.PART_OF_DOCUMENT

        qids = [qid for qid, _, _ in questions]
        golden_by_qid = {qid: _golden(golden) for qid, _, golden in questions}
        retrieved = {c: {} for c in CONFIGS}
        for cfg_name, overrides in CONFIGS.items():
            _apply(data_config, overrides)
            print(f"\n=== config {cfg_name} ({overrides}) ===")
            for qid, question, _ in questions:
                try:
                    docs = qa.search_question(question)
                    top = _ranked_documents(docs, part)
                except Exception as exc:
                    print(f"  {qid}: ERROR {exc}")
                    top = []
                retrieved[cfg_name][qid] = top
                print(f"  {qid}: {len(top)} docs")

        # comparison.csv (top-10 newline-joined) + comparison.json (lists)
        q_text = {qid: question for qid, question, _ in questions}
        rows = []
        for qid in qids:
            row = {"question_id": qid, "question": q_text[qid],
                   "golden_source": ", ".join(sorted(golden_by_qid[qid]))}
            for cfg_name in CONFIGS:
                row[f"{cfg_name}_top10"] = "\n".join(retrieved[cfg_name].get(qid, []))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False, encoding="utf-8")

        comparison_json = []
        for qid in qids:
            entry = {"question_id": qid, "question": q_text[qid],
                     "golden_source": sorted(golden_by_qid[qid])}
            for cfg_name in CONFIGS:
                entry[cfg_name] = retrieved[cfg_name].get(qid, [])
            comparison_json.append(entry)
        comp_path.write_text(
            json.dumps(comparison_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # --- metrics ------------------------------------------------------------
    metrics: dict[str, dict[str, float]] = {}
    n = len(qids) or 1
    for cfg_name in CONFIGS:
        agg = {f"hit@{k}": 0.0 for k in K_LIST}
        agg["mrr"] = 0.0
        for qid in qids:
            g = golden_by_qid[qid]
            retr = retrieved[cfg_name].get(qid, [])
            for k in K_LIST:
                agg[f"hit@{k}"] += _hit_at_k(g, retr, k)
            agg["mrr"] += _mrr(g, retr)
        metrics[cfg_name] = {key: round(val / n, 4) for key, val in agg.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    hdr = "  ".join(f"Hit@{k}" for k in K_LIST) + "  MRR"
    print(f"\nconfig        {hdr}")
    for cfg_name in CONFIGS:
        m = metrics[cfg_name]
        vals = "  ".join(f"{m[f'hit@{k}']:.2f}" for k in K_LIST)
        print(f"{cfg_name:12}  {vals}  {m['mrr']:.2f}")

    set_style()
    _plot_identified(qids, golden_by_qid, retrieved, out, n)
    _plot_metrics(metrics, out)

    print(f"\nWrote:\n  {out/'comparison.csv'}\n  {out/'comparison.json'}\n  {out/'metrics.json'}\n  {out/'plot_identified.png'}\n  {out/'plot_metrics.png'}")


def _plot_identified(qids, golden_by_qid, retrieved, out: Path, n: int) -> None:
    cfgs = list(CONFIGS)
    found = [
        int(sum(_hit_at_k(golden_by_qid[qid], retrieved[c].get(qid, []), TOP_DOCS) for qid in qids))
        for c in cfgs
    ]
    miss = [n - x for x in found]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x, found, 0.55, label="Golden document found", color=WONG["green"])
    b2 = ax.bar(x, miss, 0.55, bottom=found, label="Not found", color=WONG["vermillion"])
    ax.bar_label(b1, label_type="center", color="white", fontsize=10)
    ax.bar_label(b2, label_type="center", color="white", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in cfgs])
    ax.set_ylabel(f"Questions (of {n})")
    ax.set_title(f"Source identification (golden document in top-{TOP_DOCS})")
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "plot_identified.png")
    plt.close(fig)


def _plot_metrics(metrics: dict, out: Path) -> None:
    cfgs = list(CONFIGS)
    keys = [f"hit@{k}" for k in K_LIST] + ["mrr"]
    labels = [f"Hit@{k}" for k in K_LIST] + ["MRR"]
    x = np.arange(len(keys))
    w = 0.8 / max(1, len(cfgs))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, cfg_name in enumerate(cfgs):
        vals = [metrics[cfg_name][k] for k in keys]
        offs = x + (i - (len(cfgs) - 1) / 2) * w
        bars = ax.bar(offs, vals, w, label=LABELS.get(cfg_name, cfg_name), color=COLORS.get(cfg_name))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Graph retrieval quality by engine", pad=34)
    ax.grid(axis="y", color=WONG["grey"], alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, ncol=len(cfgs), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(out / "plot_metrics.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

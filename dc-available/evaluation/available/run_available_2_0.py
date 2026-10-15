"""AVAILABLE 2.0 final runner — one-file pipeline for a dataset.

The 2.0 production config is held fixed:
    hybrid (vector + PPR-merge graph), RERANKER OFF.

For a topic it does, in one pass:
  1. retrieve   hybrid + PPR-merge, reranker disabled, top-10 docs + contexts
  2. generate   LLM answers from the retrieved contexts (reuses generate.py)
  3. score      the 4 RAGAS metrics (reuses ragas_eval.py)

Writes to  <out_dir>  (default new_results/available2.0_results/<topic>/):
    records.json         per-question: retrieved docs/contexts, golden_rank,
                         generated_answer, retrieval_time, generation_time
    ragas_scores.json    per-row RAGAS metrics
    ragas_scores.csv
    ragas_summary.json   mean of the 4 metrics
    retrieval_metrics.json   Hit@1/3/10 + MRR (document level)

Grownfield is NOT handled here — its 2.0 == reranker_ablation `no_rerank`
(already copied into available2.0_results/grownfield/).

Usage (from project root, needs Qdrant + Azure OpenAI):
  poetry run python -m evaluation.available.run_available_2_0 hydraulics
  poetry run python -m evaluation.available.run_available_2_0 synthetic [out_dir]
"""
from __future__ import annotations

# Opik off before any dc_available import (keeps timings/logs clean).
import os

os.environ["OPIK_TRACK_DISABLE"] = "true"
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import logging

logging.getLogger("opik").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.ERROR)

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Drop the eval dirs so local evaluation/datasets can't shadow HF "datasets",
# but KEEP the project root (dc-available) so dc_available imports.
_here = Path(__file__).resolve().parent        # .../evaluation/available
_root = _here.parents[1]                        # .../dc-available
_drop = {str(_here), str(_here.parent)}         # available, evaluation
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _drop]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from evaluation.available.retrieval import common as gc
from evaluation.available.generation import generate as gen
from evaluation.available.generation import ragas_eval as rg

K_LIST = [1, 3, 10]
TOP_DOCS = 10

# The fixed 2.0 retrieval path: hybrid (vector + PPR-merge), reranker OFF.
LEG_2_0 = {
    "FUSION_LEGS": "both",
    "GRAPH_RETRIEVAL_MODE": "ppr",
    "PPR_SEED_MODE": "merge",
    "RERANKER_ENABLED": False,
}

# Per-topic dataset + columns + golden-matching strategy.
#   match "filename" -> gc golden/stem matching;  "norm" -> synthetic _norm equality.
TOPICS: dict[str, dict] = {
    "hydraulics": {
        "dataset": "evaluation/datasets/hydraluics/01_Teilnehmerunterlage_RT04-004_DE_dataset.xlsx",
        "col_question": "question", "col_golden": "source", "col_answer": "answer",
        "col_qtype": "question_type", "match": "filename",
    },
    "synthetic": {
        "dataset": "evaluation/datasets/Synthetic/generated_dataset.xlsx",
        "col_question": "question", "col_golden": "source_document", "col_answer": "answer",
        "col_qtype": "question_type", "match": "norm",
    },
}

# --- synthetic-style filename normalisation (from retrieval/synthetic.py) ----
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def _norm(value) -> str:
    s = _EXT_RE.sub("", str(value or "").strip()).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _rank_of(golden_raw, docs: list[str], match: str) -> int | None:
    """Rank (1-based) of the golden document in the retrieved list, or None."""
    if match == "norm":
        key = _norm(golden_raw)
        if not key:
            return None
        for i, name in enumerate(docs, 1):
            if _norm(name) == key:
                return i
        return None
    gset = gc.golden_of(golden_raw, match)          # "filename" / "code"
    if not gset:
        return None
    for i, name in enumerate(docs, 1):
        if gc._doc_keys(name, match) & gset:
            return i
    return None


def load_rows(cfg: dict) -> list[dict]:
    df = pd.read_excel(cfg["dataset"])
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        q = str(r.get(cfg["col_question"], "")).strip()
        if not q or q.lower() == "nan":
            continue
        rows.append({
            "question_id": f"q{i + 1}",
            "question": q,
            "question_type": str(r.get(cfg["col_qtype"], "")).strip().lower(),
            "golden_source": str(r.get(cfg["col_golden"], "")).strip(),
            "golden_answer": str(r.get(cfg["col_answer"], "")).strip(),
        })
    print(f"Loaded {len(rows)} questions from {Path(cfg['dataset']).name}")
    return rows


def retrieve_and_generate(rows: list[dict], cfg: dict) -> list[dict]:
    """One hybrid/no-reranker retrieval pass + LLM answer per question."""
    from dc_available.config import data_config
    data_config.update_data_config()
    data_config.RETRIEVAL_MODE = "fusion"
    from dc_available.RAG import qa_functions as qa
    from dc_available.RAG.data import AvailableMetaData

    PART = AvailableMetaData.PART_OF_DOCUMENT
    for key, val in LEG_2_0.items():
        setattr(data_config, key, val)
    data_config.FUSION_OUTPUT_K = TOP_DOCS

    llm = gen._build_llm()
    records = []
    for i, row in enumerate(rows, 1):
        q = row["question"]
        t0 = time.perf_counter()
        try:
            final = qa.search_question(q)
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] {row['question_id']}: RETRIEVAL ERROR {exc}")
            final = []
        retrieval_time = time.perf_counter() - t0

        # unique top-N docs (rank order) + chunk contexts
        seen, post_docs = set(), []
        for d in final:
            name = d.metadata.get(PART) or "N/A"
            if name not in seen:
                seen.add(name)
                post_docs.append(name)
            if len(post_docs) >= TOP_DOCS:
                break
        contexts = [(d.page_content or "") for d in final]

        answer, gen_time = "", 0.0
        ctx = [c for c in contexts if c]
        if ctx:
            t1 = time.perf_counter()
            try:
                answer = gen._generate(llm, q, ctx)
            except Exception as exc:
                print(f"  [{i}/{len(rows)}] {row['question_id']}: GEN ERROR {exc}")
            gen_time = time.perf_counter() - t1

        rank = _rank_of(row["golden_source"], post_docs, cfg["match"])
        records.append({
            "question_id": row["question_id"],
            "question": q,
            "question_type": row["question_type"],
            "golden_source": row["golden_source"],
            "golden_answer": row["golden_answer"],
            "retrieved_docs": post_docs,
            "retrieved_contexts": contexts,
            "generated_answer": answer,
            "golden_rank": rank,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(gen_time, 3),
        })
        print(f"  [{i}/{len(rows)}] {row['question_id']}: rank={rank} "
              f"({len(post_docs)} docs, {len(answer)} chars)")
    return records


def retrieval_metrics(records: list[dict]) -> dict:
    n = len(records) or 1
    ranks = [r["golden_rank"] for r in records]
    m = {f"hit@{k}": round(sum(1 for rk in ranks if rk and rk <= k) / n, 4) for k in K_LIST}
    m["mrr"] = round(sum(1.0 / rk for rk in ranks if rk) / n, 4)
    m["n"] = len(records)
    return m


def plot_retrieval(metrics: dict, out: Path, topic: str) -> None:
    """Single-system retrieval bar chart (Hit@1/3/10 + MRR), canonical style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    keys = [f"hit@{k}" for k in K_LIST] + ["mrr"]
    labels = [f"Hit@{k}" for k in K_LIST] + ["MRR"]
    vals = [metrics[k] for k in keys]
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00"][:len(keys)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, width=0.6, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(f"AVAILABLE 2.0 retrieval — {topic.capitalize()} (hybrid, no reranker)")
    ax.grid(axis="y", color="#999999", alpha=0.3, linewidth=0.6)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else ""
    if topic not in TOPICS:
        raise SystemExit(f"topic must be one of {list(TOPICS)} (grownfield 2.0 = reranker no_rerank).")
    cfg = TOPICS[topic]
    out_dir = (Path(sys.argv[2]) if len(sys.argv) > 2
               else Path("evaluation/new_results/available2.0_results") / topic)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== AVAILABLE 2.0 run: {topic} (hybrid + PPR, reranker OFF) -> {out_dir} ===")

    # 1-2. retrieve + generate
    rows = load_rows(cfg)
    records = retrieve_and_generate(rows, cfg)
    records_path = out_dir / "records.json"
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    # retrieval metrics + plot
    rm = retrieval_metrics(records)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(rm, indent=2), encoding="utf-8")
    plot_retrieval(rm, out_dir / "plot_retrieval.png", topic)
    print(f"\nretrieval: Hit@1={rm['hit@1']:.2f} Hit@3={rm['hit@3']:.2f} "
          f"Hit@10={rm['hit@10']:.2f} MRR={rm['mrr']:.2f} (n={rm['n']})")

    # 3. RAGAS 4 metrics (reuse the E2 scorer on records.json)
    print("\n=== RAGAS scoring ===")
    ragas_llm, ragas_emb = rg._build_ragas_llm_emb()
    rg.evaluate_topic(topic, ragas_llm, ragas_emb, in_file=records_path, out_dir=out_dir)
    # evaluate_topic writes scores.{csv,json} + summary.json + plot_ragas.png;
    # rename to the available2.0 convention (records.json + ragas_*).
    for src, dst in [("scores.json", "ragas_scores.json"),
                     ("scores.csv", "ragas_scores.csv"),
                     ("summary.json", "ragas_summary.json")]:
        sp = out_dir / src
        if sp.exists():
            sp.replace(out_dir / dst)

    print(f"\nWrote records.json + ragas_scores.json/csv + ragas_summary.json + "
          f"retrieval_metrics.json to {out_dir}")


if __name__ == "__main__":
    main()

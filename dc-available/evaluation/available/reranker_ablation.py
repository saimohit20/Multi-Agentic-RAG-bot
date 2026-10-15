"""Reranker ablation on Grownfield: no_rerank vs base vs v2m3 (current).

Single-file, self-contained quality-vs-latency study to decide whether the
cross-encoder reranker earns its ~20 s CPU cost. The production retrieval path is
held fixed (hybrid: vector + PPR-merge graph); only the reranker is varied:

    no_rerank   RERANKER_ENABLED=False          — union order only (latency floor)
    base        BAAI/bge-reranker-base          — smaller multilingual (~2x faster)
    v2m3        BAAI/bge-reranker-v2-m3          — current model (quality ceiling)

Per question x config it records, in ONE row:
    question, golden_source, golden_answer,
    retrieved_contexts, generated_answer,
    golden_rank, answer_correctness,
    retrieval_time, rerank_time, generation_time

Metrics:
    retrieval   MRR + Hit@1/@3/@10 (document level, post-rerank)
    generation  RAGAS answer_correctness (vs golden_answer)

Opik tracing is force-disabled (OPIK_TRACK_DISABLE) so span overhead does not
pollute the latency numbers; timings come from perf_counter around the real code.
The reranker model is a singleton, so it is reset between configs to force the
swap, and a warm-up question (untimed) pays the one-time model load per config.

Usage:
  poetry run python -m evaluation.available.reranker_ablation [dataset.xlsx] [out_dir]
Default out_dir: evaluation/results/reranker_ablation/grownfield/
"""
from __future__ import annotations

# --- turn Opik OFF before anything imports it (span overhead would skew timings) ---
import os

os.environ["OPIK_TRACK_DISABLE"] = "true"
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")   # don't route localhost via the proxy

# Silence noisy loggers that don't affect results: Opik's failed span uploads
# (502 via the proxy) and Streamlit's "missing ScriptRunContext" warnings (we run
# qa_functions in bare mode, not inside a Streamlit app).
import logging

logging.getLogger("opik").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Running "python evaluation/..." puts evaluation/ on sys.path, which lets the
# local "datasets" folder shadow HuggingFace's "datasets" (imported by ragas /
# FlagEmbedding). Drop the script dir + evaluation/ before any dc_available import.
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent     # .../evaluation/available
_eval_dir = _here.parent                    # .../evaluation
_root = _here.parents[1]                     # .../dc-available (holds the dc_available package)
# Drop the script + evaluation dirs so local "evaluation/datasets" can't shadow
# HuggingFace's "datasets", but KEEP the project root so dc_available imports.
_drop = {str(_here), str(_eval_dir)}
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _drop]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import io
import json
import re
import time
from contextlib import redirect_stdout

import pandas as pd

# Configure the data source BEFORE importing qa_functions (its import runs
# init_global_vars(), which loads the index / embeddings using these paths).
from dc_available.config import data_config

data_config.update_data_config()
data_config.RETRIEVAL_MODE = "fusion"

from dc_available.RAG import qa_functions as qa
from dc_available.RAG import reranker as reranker_mod
from dc_available.RAG.data import AvailableMetaData
from dc_available.RAG.qa_functions import EXAMPLE_PROMPT, search_question
from dc_available.config.llm_config import get_langchain_llm

from evaluation.available.retrieval import common as gc

from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# --- what we vary -----------------------------------------------------------
# The production retrieval path is held constant; only the reranker changes.
BASE_LEG = {"FUSION_LEGS": "both", "GRAPH_RETRIEVAL_MODE": "ppr", "PPR_SEED_MODE": "merge"}

CONFIGS: dict[str, dict] = {
    "no_rerank": {"RERANKER_ENABLED": False},
    "base":      {"RERANKER_ENABLED": True, "RERANKER_MODEL": "BAAI/bge-reranker-base"},
    "v2m3":      {"RERANKER_ENABLED": True, "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3"},
}

DATASET = Path("evaluation/datasets/Grownfield/dataset_full.xlsx")
MATCH_MODE = "code"          # golden_codes like O2C-02.010
TOP_DOCS = gc.TOP_DOCS       # 10
K_LIST = gc.K_LIST           # [1, 3, 10]

PART = AvailableMetaData.PART_OF_DOCUMENT


# --- rerank-time capture ----------------------------------------------------
# qa_functions calls reranker.rerank via the name `_cross_encoder_rerank`. Wrap it
# so we time ONLY the cross-encoder, independent of vector/graph. Reset per question.
_last_rerank_time = 0.0
_orig_rerank = qa._cross_encoder_rerank


def _timed_rerank(*args, **kwargs):
    global _last_rerank_time
    t0 = time.perf_counter()
    out = _orig_rerank(*args, **kwargs)
    _last_rerank_time = time.perf_counter() - t0
    return out


qa._cross_encoder_rerank = _timed_rerank

# qa_functions prints its own per-query breakdown:
#   "[LAT] retrieval breakdown  vector=1.23s  graph=4.56s  rerank=7.89s"
# We reuse the runtime's own perf_counter numbers for vector + graph (the single
# source of truth) by parsing that line; rerank comes from the wrapper above.
_LAT_RE = re.compile(r"vector=([\d.]+)s\s+graph=([\d.]+)s\s+rerank=([\d.]+)s")


def _timed_search(question: str):
    """Run search_question, returning (documents, total, vector_time, graph_time).

    stdout is captured so the voluminous per-chunk debug is suppressed and the
    [LAT] line can be parsed for the vector/graph split.
    """
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        documents = search_question(question)
    total = time.perf_counter() - t0
    m = _LAT_RE.search(buf.getvalue())
    vector_time, graph_time = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    return documents, total, vector_time, graph_time


# --- dataset ----------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    df = pd.read_excel(path) if str(path).lower().endswith(".xlsx") else pd.read_csv(path)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        rows.append({
            "question_id": str(r.get("question_id", f"q{i + 1}")),
            "question": str(r.get("question", "")).strip(),
            "golden_answer": str(r.get("golden_answer", "")).strip(),
            "golden_source": sorted(gc.golden_of(r.get("golden_codes"), MATCH_MODE)),
            "_golden": gc.golden_of(r.get("golden_codes"), MATCH_MODE),   # set, for matching
        })
    return rows


# --- helpers ----------------------------------------------------------------
def _unique_docs(documents, k: int) -> list[str]:
    seen, out = set(), []
    for d in documents:
        name = d.metadata.get(PART) or "N/A"
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= k:
            break
    return out


def _rank_of(golden: set[str], docs: list[str]) -> int | None:
    for i, name in enumerate(docs, 1):
        if gc._doc_keys(name, MATCH_MODE) & golden:
            return i
    return None


def _apply(overrides: dict) -> None:
    for key, value in overrides.items():
        setattr(data_config, key, value)


def _generate(llm_chain, documents, question: str) -> str:
    """Run the same LangChain QA chain used in production (see qa_runner)."""
    response = llm_chain(
        {"input_documents": documents, "question": question},
        return_only_outputs=True,
    )
    return response["text"].strip()


# --- one config -------------------------------------------------------------
def run_config(name: str, overrides: dict, rows: list[dict], llm_chain) -> list[dict]:
    global _last_rerank_time

    _apply(BASE_LEG)
    _apply(overrides)
    data_config.FUSION_OUTPUT_K = TOP_DOCS
    # Force the reranker singleton to reload so the model actually swaps per config.
    reranker_mod._reranker = None

    print(f"\n=== config: {name} ({overrides}) ===")

    # Warm-up (untimed): pays the one-time model load so the first real timing is clean.
    if rows:
        try:
            search_question(rows[0]["question"])
        except Exception as exc:
            print(f"  [warmup] {exc}")

    records = []
    for row in rows:
        q = row["question"]
        _last_rerank_time = 0.0
        try:
            documents, retrieval_time, vector_time, graph_time = _timed_search(q)
            rerank_time = _last_rerank_time          # 0.0 when RERANKER_ENABLED=False

            t1 = time.perf_counter()
            answer = _generate(llm_chain, documents, q)
            generation_time = time.perf_counter() - t1

            contexts = [(d.page_content or "") for d in documents]
            post_docs = _unique_docs(documents, TOP_DOCS)
            golden_rank = _rank_of(row["_golden"], post_docs)
        except Exception as exc:
            print(f"  {row['question_id']}: ERROR {exc}")
            documents, contexts, post_docs = [], [], []
            answer, golden_rank = "", None
            retrieval_time = vector_time = graph_time = rerank_time = generation_time = 0.0

        records.append({
            "question_id": row["question_id"],
            "question": q,
            "golden_source": row["golden_source"],
            "golden_answer": row["golden_answer"],
            "retrieved_docs": post_docs,
            "retrieved_contexts": contexts,
            "generated_answer": answer,
            "golden_rank": golden_rank,
            "answer_correctness": None,              # filled by RAGAS below
            "vector_time": round(vector_time, 3),
            "graph_time": round(graph_time, 3),
            "rerank_time": round(rerank_time, 3),
            "generation_time": round(generation_time, 3),
            "retrieval_time": round(retrieval_time, 3),   # total search_question wall time
        })
        print(f"  {row['question_id']}: rank={golden_rank}  "
              f"vec={vector_time:.2f}s graph={graph_time:.2f}s "
              f"rerank={rerank_time:.2f}s gen={generation_time:.2f}s")

    return records


# --- RAGAS answer_correctness ----------------------------------------------
def score_answer_correctness(records: list[dict]) -> None:
    """Compute RAGAS answer_correctness and merge it into each record in place."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_correctness
    from dc_available.config.data_config import DEFAULT_EMBEDDING_MODEL
    from dc_available.RAG.retrieval_helpers import get_embeddings

    valid_idx = [
        i for i, r in enumerate(records)
        if r["generated_answer"] and r["retrieved_contexts"] and r["golden_answer"]
    ]
    if not valid_idx:
        print("  [ragas] no valid rows to score.")
        return

    data = {
        "question": [records[i]["question"] for i in valid_idx],
        "answer": [records[i]["generated_answer"] for i in valid_idx],
        "contexts": [records[i]["retrieved_contexts"] for i in valid_idx],
        "ground_truth": [records[i]["golden_answer"] for i in valid_idx],
    }
    llm = get_langchain_llm(model_name="gpt-4o")
    embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
    result = evaluate(
        Dataset.from_dict(data),
        metrics=[answer_correctness],
        llm=llm,
        embeddings=embeddings,
    )
    scores = result.to_pandas()["answer_correctness"].tolist()
    for j, i in enumerate(valid_idx):
        records[i]["answer_correctness"] = round(float(scores[j]), 4) if scores[j] == scores[j] else None


# --- metrics summary --------------------------------------------------------
def summarize(name: str, records: list[dict]) -> dict:
    n = len(records) or 1
    ranked = [r["golden_rank"] for r in records if r["golden_rank"]]
    ac = [r["answer_correctness"] for r in records if r["answer_correctness"] is not None]
    # Mean rerank time over questions where it actually ran.
    rr = [r["rerank_time"] for r in records if r["rerank_time"] > 0]
    row = {"config": name}
    for k in K_LIST:
        row[f"hit@{k}"] = round(sum(1 for rk in ranked if rk <= k) / n, 4)
    row["mrr"] = round(sum(1.0 / rk for rk in ranked) / n, 4)
    row["answer_correctness"] = round(sum(ac) / len(ac), 4) if ac else None
    row["mean_vector_time"] = round(sum(r["vector_time"] for r in records) / n, 3)
    row["mean_graph_time"] = round(sum(r["graph_time"] for r in records) / n, 3)
    row["mean_rerank_time"] = round(sum(rr) / len(rr), 3) if rr else 0.0
    row["mean_generation_time"] = round(sum(r["generation_time"] for r in records) / n, 3)
    row["mean_retrieval_time"] = round(sum(r["retrieval_time"] for r in records) / n, 3)
    return row


# --- orchestrator -----------------------------------------------------------
def main() -> None:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET
    out_dir = (Path(sys.argv[2]) if len(sys.argv) > 2
               else Path("evaluation/results/reranker_ablation/grownfield"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset)
    print(f"Loaded {len(rows)} questions from {dataset}")

    # Build the generation chain once (same chain production uses).
    llm = get_langchain_llm()
    llm_chain = load_qa_with_sources_chain(
        llm, chain_type="stuff", output_key="text", verbose=False,
        document_prompt=EXAMPLE_PROMPT,
    )

    summary_rows = []
    for name, overrides in CONFIGS.items():
        records = run_config(name, overrides, rows, llm_chain)
        print(f"  [ragas] scoring answer_correctness for {name} ...")
        score_answer_correctness(records)
        (out_dir / f"{name}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_rows.append(summarize(name, records))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    print("\n=== Quality vs latency ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote per-config JSONs + summary.csv/json to {out_dir}")


if __name__ == "__main__":
    main()

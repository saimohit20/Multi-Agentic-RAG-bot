"""Synthetic retrieval evaluation — vector / graph / hybrid legs, all questions.

Runs the three retrieval legs over the full synthetic dataset and records, at
DOCUMENT level, where the golden source_document lands per leg (post-rerank and
pre-rerank pool). No plots — JSON only.

Writes to evaluation/results/synthetic/ :
  retrieval.json          one record per (question x leg)  -> n_questions x 3
  retrieval_metrics.json  Hit@1/3/10 + MRR per leg (document level)

Run from the project root (dc-available):
  poetry run python -m evaluation.available.retrieval.synthetic
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

DATASET = Path("evaluation/datasets/Synthetic/generated_dataset.xlsx")
OUT_DIR = Path("evaluation/results/synthetic")

# leg -> data_config overrides (same knobs the grownfield/hydraulics evals use)
LEGS = {
    "vector": {"FUSION_LEGS": "vector"},
    "graph":  {"FUSION_LEGS": "graph", "GRAPH_RETRIEVAL_MODE": "ppr", "PPR_SEED_MODE": "merge"},
    "hybrid": {"FUSION_LEGS": "both",  "GRAPH_RETRIEVAL_MODE": "ppr", "PPR_SEED_MODE": "merge"},
}
K_LIST = [1, 3, 10]
TOP_DOCS = 10


_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def _norm(value) -> str:
    """Comparable key for a document name. Drop a trailing FILE extension only
    (not mid-name dots), lower-case, and collapse every run of non-alphanumerics
    (spaces, '_', '-', '.') to a single space. Robust to capitals, underscores,
    dashes and mid-name dots.
    'B2P-01.020 GUI Subs. Plan. - Sim.pdf' -> 'b2p 01 020 gui subs plan sim'."""
    s = _EXT_RE.sub("", str(value or "").strip()).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def load_rows(path: Path) -> list[dict]:
    df = pd.read_excel(path)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        q = str(r.get("question", "")).strip()
        if not q or q.lower() == "nan":
            continue
        rows.append({
            "question_id": f"q{i + 1}",
            "question": q,
            "topic": str(r.get("topic", "")).strip(),
            "level": str(r.get("level", "")).strip(),
            "type": str(r.get("type", "")).strip(),
            "question_type": str(r.get("question_type", "")).strip().lower(),
            "golden_source": str(r.get("source_document", "")).strip(),
            "golden_answer": str(r.get("answer", "")).strip(),
        })
    print(f"Loaded {len(rows)} questions from {path.name}")
    return rows


def _unique_docs(chunks, PART, k: int) -> list[str]:
    seen, out = set(), []
    for d in chunks:
        doc = d.metadata.get(PART) or "N/A"
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
        if len(out) >= k:
            break
    return out


def _rank_of(golden: str, docs: list[str]) -> int | None:
    key = _norm(golden)
    if not key:
        return None
    for i, name in enumerate(docs, 1):
        if _norm(name) == key:
            return i
    return None


def run_retrieval(rows: list[dict]) -> list[dict]:
    # Drop this package's dirs from sys.path so the local "datasets" folder can't
    # shadow HuggingFace's "datasets" imported deep inside the RAG stack.
    here = Path(__file__).resolve().parent                       # .../src/synthetic
    bad = {str(here), str(here.parents[0]), str(here.parents[1])}  # synthetic, src, evaluation
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in bad]

    from dc_available.config import data_config
    data_config.update_data_config()
    data_config.RETRIEVAL_MODE = "fusion"
    from dc_available.RAG import qa_functions as qa
    from dc_available.RAG.data import AvailableMetaData

    PART = AvailableMetaData.PART_OF_DOCUMENT
    records = []

    for leg, overrides in LEGS.items():
        print(f"\n=== leg: {leg} ({overrides}) ===")
        for row in rows:
            q = row["question"]
            golden = row["golden_source"]
            for key, val in overrides.items():
                setattr(data_config, key, val)

            # retrieval — the real (post-rerank) answer path, one pass per question
            data_config.RERANKER_ENABLED = True
            data_config.FUSION_OUTPUT_K = TOP_DOCS
            try:
                final = qa.search_question(q)
            except Exception as exc:
                print(f"  {row['question_id']}/{leg}: ERROR {exc}")
                final = []
            post_docs = _unique_docs(final, PART, TOP_DOCS)
            contexts = [(d.page_content or "") for d in final]

            rec = {
                "question_id": row["question_id"],
                "question": q,
                "topic": row["topic"],
                "level": row["level"],
                "type": row["type"],
                "question_type": row["question_type"],
                "golden_source": golden,
                "golden_answer": row["golden_answer"],
                "retrieval_type": leg,
                "retrieved_docs": post_docs,
                "golden_rank": _rank_of(golden, post_docs),
                "retrieved_contexts": contexts,
            }
            records.append(rec)
            print(f"  {row['question_id']}/{leg}: rank={rec['golden_rank']} ({len(post_docs)} docs)")

    return records


def compute_metrics(records: list[dict]) -> dict:
    metrics = {}
    for leg in LEGS:
        recs = [r for r in records if r["retrieval_type"] == leg]
        n = len(recs) or 1
        m = {f"hit@{k}": round(sum(1 for r in recs
                                   if r["golden_rank"] and r["golden_rank"] <= k) / n, 4)
             for k in K_LIST}
        m["mrr"] = round(sum(1.0 / r["golden_rank"] for r in recs if r["golden_rank"]) / n, 4)
        m["n"] = len(recs)
        metrics[leg] = m
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATASET)
    records = run_retrieval(rows)

    # one file per leg (retrieval_metrics.json stays combined)
    for leg in LEGS:
        leg_rows = [r for r in records if r["retrieval_type"] == leg]
        (OUT_DIR / f"{leg}.json").write_text(
            json.dumps(leg_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = compute_metrics(records)
    (OUT_DIR / "retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nleg      Hit@1 Hit@3 Hit@10  MRR")
    for leg in LEGS:
        m = metrics[leg]
        print(f"{leg:8} {m['hit@1']:.2f}  {m['hit@3']:.2f}  {m['hit@10']:.2f}  {m['mrr']:.2f}")
    print(f"\nWrote {', '.join(f'{leg}.json' for leg in LEGS)} and "
          f"retrieval_metrics.json to {OUT_DIR}")


if __name__ == "__main__":
    main()

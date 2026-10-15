"""
Run retrieval evaluation for the SYNTHETIC dataset.

    .venv/Scripts/python.exe evaluation/retrieval_eval/src/run_synthetic.py

Produces: evaluation/retrieval_eval/results/synthetic_results.json

Golden source is a single `source_document` name, so we reuse the Hydraulics
document-level matcher (stem comparison).
"""
from __future__ import annotations

import json
from typing import Any, Dict

import paths_config as cfg
import data_prep
import retriever_api
import eval_metrics
from hit_matching import golden_rank_synthetic


def build_record(q: Dict[str, Any]) -> Dict[str, Any]:
    retrieved = retriever_api.search(q["question"], k=cfg.TOP_K)
    rank = golden_rank_synthetic(q["golden_source"], retrieved)
    return {
        "question_id": q["question_id"],
        "question": q["question"],
        "question_type": q["question_type"],
        "level": q["level"],
        "type": q["type"],
        "golden_source": q["golden_source"],          # single document name
        "golden_answer": q["golden_answer"],
        "retrieved_docs": [r["doc_name"] for r in retrieved],
        "golden_rank": rank,
        "retrieved_contexts": [r["text"] for r in retrieved],
        "generated_answer": None,
        "topic": q.get("topic"),
        "keywords": q.get("keywords"),
    }


def main() -> None:
    cfg.ensure_dirs()
    questions = data_prep.load_synthetic()
    print(f"[Synthetic] {len(questions)} questions loaded.")

    records = []
    for i, q in enumerate(questions, start=1):
        rec = build_record(q)
        rank = rec["golden_rank"]
        print(f"[Synthetic] {i:>4}/{len(questions)} {q['question_id']}  golden_rank={rank}")
        records.append(rec)

    with open(cfg.SYNTHETIC_RESULTS, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved -> {cfg.SYNTHETIC_RESULTS}")
    print(eval_metrics.format_summary("Synthetic", eval_metrics.compute(records)))


if __name__ == "__main__":
    main()

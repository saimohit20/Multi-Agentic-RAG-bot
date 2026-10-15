"""
Score generated answers with RAGAS 0.1.14.

Reads:  evaluation/generation_eval/results/{grownfield,hydraulics}_gen_results.json
Writes: same files updated in-place with per-question RAGAS scores
        evaluation/generation_eval/results/ragas_summary.json

Run:
    .venv/Scripts/python.exe evaluation/generation_eval/src/run_ragas.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import gen_paths_config as cfg

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from dc_available.config.llm_config import get_langchain_llm
from dc_available.RAG.retrieval_helpers import get_embeddings
from dc_available.config.data_config import DEFAULT_EMBEDDING_MODEL

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_dict({
        "question":     [r["question"] for r in records],
        "answer":       [r.get("generated_answer") or "" for r in records],
        "contexts":     [r["retrieved_contexts"] for r in records],
        "ground_truth": [r.get("golden_answer") or "" for r in records],
    })


def score_topic(
    name: str,
    path: Path,
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
) -> Dict[str, float]:
    if not path.exists():
        print(f"[{name}] Not found: {path}  (run generate_answers.py first)")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    missing = [r["question_id"] for r in records if not r.get("generated_answer")]
    if missing:
        print(f"[{name}] WARNING: {len(missing)} records have no generated_answer — skipping them.")
        records = [r for r in records if r.get("generated_answer")]

    if not records:
        print(f"[{name}] No scoreable records. Skipping.")
        return {}

    print(f"[{name}] Scoring {len(records)} records with RAGAS …")
    dataset = _build_dataset(records)

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df = result.to_pandas()
    for i, rec in enumerate(records):
        for metric in cfg.RAGAS_METRICS:
            rec[metric] = float(df.iloc[i][metric]) if metric in df.columns else None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[{name}] Updated -> {path}")

    aggregate = {m: float(result[m]) for m in cfg.RAGAS_METRICS if m in result}
    parts = ", ".join(f"{k}={v:.3f}" for k, v in aggregate.items())
    print(f"[{name}] {parts}")
    return aggregate


# topic key -> (display name, generation results JSON)
TOPICS = {
    "grownfield": ("Grownfield", cfg.GROWNFIELD_GEN_RESULTS),
    "hydraulics": ("Hydraulics", cfg.HYDRAULICS_GEN_RESULTS),
    "synthetic":  ("Synthetic",  cfg.SYNTHETIC_GEN_RESULTS),
}


def main(topics: List[str] | None = None) -> None:
    cfg.ensure_dirs()
    keys = topics or list(TOPICS)

    ragas_llm = LangchainLLMWrapper(get_langchain_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings(DEFAULT_EMBEDDING_MODEL))

    # Load the existing combined summary so topics we don't re-score are preserved.
    combined: Dict[str, Dict[str, float]] = {}
    if cfg.RAGAS_SUMMARY.exists():
        with open(cfg.RAGAS_SUMMARY, "r", encoding="utf-8") as f:
            combined = json.load(f)

    for n, key in enumerate(keys, start=1):
        name, path = TOPICS[key]
        print(f"\n========== {n}/{len(keys)}  {name.upper()} ==========")
        scores = score_topic(name, path, ragas_llm, ragas_emb)

        # One summary file per topic, e.g. synthetic_ragas_summary.json
        out = cfg.ragas_summary_path(key)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({key: scores}, f, ensure_ascii=False, indent=2)
        print(f"[{name}] Saved -> {out}")

        # Accumulate into the combined summary that the plots read.
        combined[key] = scores

    with open(cfg.RAGAS_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n[Summary] Saved combined -> {cfg.RAGAS_SUMMARY}")

    print("\n[Done] RAGAS scoring complete.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", choices=[*TOPICS, "all"], default="all")
    args = ap.parse_args()
    main(None if args.topic == "all" else [args.topic])

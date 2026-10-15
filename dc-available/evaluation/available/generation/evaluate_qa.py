from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
)
from dc_available.config.llm_config import get_langchain_llm
from dc_available.config.data_config import DEFAULT_EMBEDDING_MODEL
from dc_available.RAG.retrieval_helpers import get_embeddings

logger = logging.getLogger(__name__)

# Metrics that require only question + answer + contexts (no ground_truth needed)
REFERENCE_FREE_METRICS = [faithfulness, answer_relevancy, context_precision]

# Metrics that also require ground_truth
REFERENCE_METRICS = [answer_correctness]

ALL_METRICS = REFERENCE_FREE_METRICS + REFERENCE_METRICS


def run_ragas_evaluation(results: list[dict], output_dir: Path) -> dict:
    """Run RAGAS evaluation on collected QA results and save scores.

    Args:
        results: output from run_qa_dataset() — list of dicts with
                 keys: question, answer, contexts, ground_truth
        output_dir: directory to write qa_scores.json and qa_scores.xlsx

    Returns:
        dict of metric_name -> aggregate score
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop rows that errored during retrieval/generation
    valid = [
        r for r in results
        if not r.get("error") and r.get("answer") and r.get("contexts")
    ]
    skipped = len(results) - len(valid)
    if skipped:
        logger.warning("Skipping %d errored/empty results from RAGAS evaluation.", skipped)

    if not valid:
        raise ValueError("No valid results to evaluate. Check qa_intermediate.json for errors.")

    print(f"\n=== Phase 2: RAGAS Evaluation ===")

    # Determine which metrics to use based on ground_truth availability
    has_ground_truth = any(r.get("ground_truth") for r in valid)
    metrics = ALL_METRICS if has_ground_truth else REFERENCE_FREE_METRICS
    if not has_ground_truth:
        logger.warning("No ground_truth found — running reference-free metrics only.")

    data = {
        "question": [r["question"] for r in valid],
        "answer": [r["answer"] for r in valid],
        "contexts": [r["contexts"] for r in valid],
        "ground_truth": [r.get("ground_truth", "") for r in valid],
    }

    print(f"\nRunning RAGAS on {len(valid)} samples with metrics: {[m.name for m in metrics]}")
    dataset = Dataset.from_dict(data)
    llm = get_langchain_llm(model_name="gpt-4o")
    embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
    ragas_result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    print("RAGAS scoring completed")

    # Per-sample scores
    scores_df = ragas_result.to_pandas()
    scores_df.to_excel(output_dir / "qa_scores.xlsx", index=False, engine="openpyxl")

    per_sample = scores_df.to_dict(orient="records")
    # For JSON export, exclude 'contexts' (list of text chunks — too large, keep in xlsx only)
    per_sample_json = [
        {k: v for k, v in row.items() if k != "contexts"}
        for row in per_sample
    ]
    with open(output_dir / "qa_scores.json", "w", encoding="utf-8") as f:
        json.dump(per_sample_json, f, indent=2, ensure_ascii=False)

    # Combine Phase 1 (intermediate) + Phase 2 (RAGAS scores) into one xlsx
    combined_records = []
    for i, result in enumerate(valid):
        if i < len(per_sample):
            ragas_row = per_sample[i]
            combined = {
                # Phase 1 fields
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "retrieved_source": result.get("retrieved_source", []),
                "ground_truth": result["ground_truth"],
                "golden_source": result.get("golden_source", ""),
                "source_hit": result.get("source_hit"),
                "source_hit_rank": result.get("source_hit_rank"),
                "source_matches": result.get("source_matches"),
                # Phase 2 (RAGAS) fields
                "faithfulness": ragas_row.get("faithfulness"),
                "answer_relevancy": ragas_row.get("answer_relevancy"),
                "context_precision": ragas_row.get("context_precision"),
                "answer_correctness": ragas_row.get("answer_correctness"),
            }
            combined_records.append(combined)

    import pandas as pd
    combined_df = pd.DataFrame(combined_records)
    combined_df.to_excel(output_dir / "qa_results_combined.xlsx", index=False, engine="openpyxl")
    print(f"Phase 2 completed → {output_dir / 'qa_results_combined.xlsx'}\n")
    print(f"✅ Success, combined results saved\n")

    # Aggregate summary
    summary = {m.name: round(float(ragas_result[m.name]), 4) for m in metrics}
    summary["num_samples"] = len(valid)
    summary["num_skipped"] = skipped

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for name, score in summary.items():
        print(f"  {name}: {score}")

    return summary

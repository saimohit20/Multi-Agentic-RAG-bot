from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Importing qa_functions triggers init_global_vars() at module level,
# which loads the vector index (Qdrant), embeddings, and LLM into global state.
from dc_available.RAG.qa_functions import search_question, EXAMPLE_PROMPT
from dc_available.config.llm_config import get_langchain_llm
from dc_available.RAG.data import AvailableMetaData

logger = logging.getLogger(__name__)


def _normalize(value: str) -> str:
    """Lowercase, unify path separators, strip — for substring source matching."""
    return str(value or "").replace("\\", "/").lower().strip()


def _doc_source(doc) -> str:
    """The storage path (contains the filename) of a retrieved chunk; falls back to 'source'."""
    return doc.metadata.get(AvailableMetaData.STORAGE_PATH) or doc.metadata.get("source", "") or ""


def _source_match(golden_source: str, retrieved_sources: list[str]) -> dict:
    """Deterministic substring check: is the golden module part of any retrieved source path?

    Returns source_hit (bool/None), source_hit_rank (1-based index of first match or None),
    and source_matches (count of retrieved chunks that match). None when golden_source is empty.
    """
    golden = _normalize(golden_source)
    if not golden:
        return {"source_hit": None, "source_hit_rank": None, "source_matches": 0}
    matched_ranks = [
        rank for rank, src in enumerate(retrieved_sources, 1)
        if golden in _normalize(src)
    ]
    return {
        "source_hit": len(matched_ranks) > 0,
        "source_hit_rank": matched_ranks[0] if matched_ranks else None,
        "source_matches": len(matched_ranks),
    }


def run_single_question(question: str) -> dict:
    """Run the QA pipeline for one question without any Streamlit dependency.

    Replicates the retrieval + generation logic of ask_question() but returns
    raw answer text and context strings instead of rendering to Streamlit.

    Returns:
        dict with keys: question, answer, contexts (list[str]), retrieved_source (list[str])
    """
    # Step 1: retrieve relevant documents via vector search + graph expansion
    documents = search_question(question)

    # Step 2: extract raw text + source path from each document
    contexts = [doc.page_content for doc in documents]
    retrieved_source = [_doc_source(doc) for doc in documents]

    # Step 3: run the same LangChain QA chain used in production
    llm = get_langchain_llm()
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        output_key="text",
        verbose=False,
        document_prompt=EXAMPLE_PROMPT,
    )
    response = chain(
        {"input_documents": documents, "question": question},
        return_only_outputs=True,
    )
    answer = response["text"].strip()

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "retrieved_source": retrieved_source,
    }


def run_qa_dataset(dataset: list[dict], output_dir: Path) -> list[dict]:
    """Run all questions through the QA pipeline and save intermediate results.

    Saves after every question so a crash does not lose completed work.
    On re-run, pass the same output_dir to resume from where it left off.

    Args:
        dataset: list of dicts with keys: question, ground_truth, golden_source
        output_dir: directory to write qa_intermediate.json

    Returns:
        list of dicts with keys: question, answer, contexts, retrieved_source,
        ground_truth, golden_source, source_hit, source_hit_rank, source_matches
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_file = output_dir / "qa_intermediate.json"

    # Resume from existing intermediate file if present
    results: list[dict] = []
    completed_questions: set[str] = set()
    if intermediate_file.exists():
        with open(intermediate_file) as f:
            results = json.load(f)
        completed_questions = {r["question"] for r in results}
        print(f"Resuming: {len(completed_questions)} questions already completed.")

    remaining = [item for item in dataset if item["question"] not in completed_questions]
    total = len(dataset)

    print(f"\n=== Phase 1: Generate Predictions ===")
    print(f"Total questions: {total}")

    for item in remaining:
        current = len(results) + 1
        print(f"[{current}/{total}] {item['question'][:70]}...")

        try:
            result = run_single_question(item["question"])
            result["ground_truth"] = item["ground_truth"]
            result["golden_source"] = item.get("golden_source", "")
            result.update(_source_match(result["golden_source"], result["retrieved_source"]))
        except Exception as exc:
            logger.warning("Failed on question %r: %s", item["question"], exc)
            result = {
                "question": item["question"],
                "answer": "",
                "contexts": [],
                "retrieved_source": [],
                "ground_truth": item["ground_truth"],
                "golden_source": item.get("golden_source", ""),
                "source_hit": None,
                "source_hit_rank": None,
                "source_matches": 0,
                "error": str(exc),
            }

        results.append(result)

        # Save after every question
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    scored = [r for r in results if r.get("source_hit") is not None]
    success = len([r for r in results if not r.get("error")])
    failed = len(results) - success

    print(f"  Success: {success}")
    print(f"  Failed: {failed}")

    if scored:
        hits = sum(1 for r in scored if r["source_hit"])
        print(f"  Source match: {hits}/{len(scored)} = {hits / len(scored):.1%}")

    print(f"Phase 1 completed → {intermediate_file}\n")
    return results

"""Step 6 — the 3-arm A/B ablation (retrieval + generation capture).

Same questions + same isolated index, only the trust config changes:
    A0 off    — TRUST_ENABLED=False
    A1 prior  — TRUST_ENABLED=True, TRUST_FILE=test_trust_prior.jsonl
    A2 full   — TRUST_ENABLED=True, TRUST_FILE=test_trust_full.jsonl  (prior+corroboration)

Retrieval is vector-only (FUSION_LEGS="vector") since the test corpus has no graph.

For each (question, arm) we record ONE row into that arm's file (ab_off / ab_prior /
ab_full.jsonl). Each row is both the DEBUG view and the RAGAS input:
  - docs     : document-level roll-up (unique parents, ranked, best chunk each)
  - chunks   : chunk-level detail  chunk_id | part | rerank | trust | final
  - good_doc : did the good doc rank ABOVE its twin  (retrieval metric, deterministic)
  - answer / ground_truth / contexts : the fields RAGAS consumes (no string match here;
    generation is scored later by score_ragas.py)

Resumable per (question, arm). Pass an integer to limit the questions (dry-run).

RUN (from the dc-available directory, after build_index; Qdrant up):
    python -m evaluation.trust_score.rag_ab.run_ab 6     # dry-run: first 6 questions
    python -m evaluation.trust_score.rag_ab.run_ab       # full ~190
"""
from __future__ import annotations

import json
import os

from evaluation.trust_score.rag_ab.config import (
    AB_ARM_FILES,
    DATASET_FILE,
    INDEX_CACHE,
    TEST_COLLECTION,
)

# Point the RAG at the isolated test collection BEFORE importing it.
os.environ["QDRANT_COLLECTION"] = TEST_COLLECTION

import dc_available.config.data_config as data_config          # noqa: E402
import dc_available.RAG.retrieval_helpers as rh                # noqa: E402
from dc_available.RAG.data import AvailableMetaData            # noqa: E402

TEST_TRUST_PRIOR = INDEX_CACHE.parent / "test_trust_prior.jsonl"
TEST_TRUST_FULL = INDEX_CACHE.parent / "test_trust_full.jsonl"

ARMS = [
    ("off", False, None),
    ("prior", True, str(TEST_TRUST_PRIOR)),
    ("full", True, str(TEST_TRUST_FULL)),
]


def _part(doc) -> str:
    return str(doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT, "")).strip().lower()


def _num(meta, key):
    """Metadata number as float, or None (the off arm leaves trust fields unset)."""
    v = meta.get(key)
    return round(float(v), 4) if v is not None else None


def _chunk_rows(docs) -> list[dict]:
    """Chunk-level detail in retrieval order: chunk_id | part | rerank | trust | final."""
    return [
        {
            "chunk_id": d.metadata.get(AvailableMetaData.ID),
            "part": _part(d),
            "rerank": _num(d.metadata, "rerank_score"),
            "trust": _num(d.metadata, "final_trust"),
            "final": _num(d.metadata, "final_score"),
        }
        for d in docs
    ]


def _doc_rows(docs) -> list[dict]:
    """Document-level roll-up: unique parents (first-seen = best chunk), ranked."""
    seen, rows = set(), []
    for d in docs:
        part = _part(d)
        if part in seen:
            continue
        seen.add(part)
        rows.append({"doc": part,
                     "final": _num(d.metadata, "final_score"),
                     "trust": _num(d.metadata, "final_trust")})
    return rows


def _load_done() -> set:
    done = set()
    for arm, path in AB_ARM_FILES.items():
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                done.add((json.loads(line)["question_id"], arm))
    return done


def run(limit: int | None = None) -> None:
    import dc_available.RAG.qa_functions as qa

    # Rebind the RAG globals to the isolated test index (collection + docstore).
    rh.retriever = None
    qa.retriever = rh.get_retriever(cache_path=str(INDEX_CACHE))
    qa.vectorstore = qa.retriever.vectorstore
    qa._child_docs_cache = None
    data_config.FUSION_LEGS = "vector"          # test corpus has no graph

    questions = [json.loads(line) for line in DATASET_FILE.open(encoding="utf-8")]
    if limit:
        questions = questions[:limit]
    done = _load_done()
    print(f"{len(questions)} questions x {len(ARMS)} arms "
          f"(collection={TEST_COLLECTION}, vector-only)")

    for arm, enabled, trust_file in ARMS:
        data_config.TRUST_ENABLED = enabled
        if trust_file:
            os.environ["TRUST_FILE"] = trust_file
        else:
            os.environ.pop("TRUST_FILE", None)
        out_path = AB_ARM_FILES[arm]
        print(f"\n=== arm '{arm}' (trust_enabled={enabled}) -> {out_path.name} ===", flush=True)

        for i, q in enumerate(questions, 1):
            qid = q["question_id"]
            if (qid, arm) in done:
                continue
            docs = qa.search_question(q["question"], return_outputs=False)
            parts = [_part(d) for d in docs]
            good_id, twin_id = q["good_doc_id"].lower(), q["twin_doc_id"].lower()
            good_rank = next((k for k, p in enumerate(parts) if p == good_id), None)
            twin_rank = next((k for k, p in enumerate(parts) if p == twin_id), None)
            good_doc = (good_rank is not None
                        and (twin_rank is None or good_rank < twin_rank))

            answer, _ = qa._generate_structured_answer(q["question"], docs)

            row = {
                "question_id": qid, "arm": arm,
                "question": q["question"],
                "lever": q["lever"],
                "good_doc_id": q["good_doc_id"], "twin_doc_id": q["twin_doc_id"],
                "docs": _doc_rows(docs),
                "chunks": _chunk_rows(docs),
                "good_rank": good_rank, "twin_rank": twin_rank,
                "good_doc": good_doc,
                "answer": (answer or "").strip(),
                "ground_truth": q.get("gold_answer", ""),
                "correct_value": q.get("correct_value", ""),
                "wrong_value": q.get("wrong_value", ""),
                "contexts": [d.page_content for d in docs],
            }
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  {arm}: {i}/{len(questions)}", flush=True)

    print("\n-> " + ", ".join(AB_ARM_FILES[a].name for a, *_ in ARMS))


if __name__ == "__main__":
    import sys
    run(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)

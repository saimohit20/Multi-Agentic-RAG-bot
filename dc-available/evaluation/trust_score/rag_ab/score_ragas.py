"""Step 7 — score generation quality with RAGAS answer_correctness.

Reads the per-arm files written by run_ab.py (ab_off / ab_prior / ab_full.jsonl) and,
for each (question, arm), computes RAGAS ``answer_correctness`` — the ONLY generation
metric (string match was dropped). We use FACTUAL-ONLY weights ``[1.0, 0.0]`` so the
score is the statement-presence F1 of the answer vs the gold answer (an LLM judge that
tolerates paraphrase) and needs NO embeddings deployment.

Judge = ``get_langchain_llm("gpt-4o")`` (temp 0, seed 42 → deterministic).

Kept separate from run_ab.py so re-scoring never re-runs retrieval/generation.
Resumable per (question_id, arm). Pass an integer to limit questions (dry-run).

Output: ab_scores.jsonl — rows ``{question_id, arm, answer_correctness}``.

RUN (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.score_ragas 6     # dry-run: first 6 per arm
    python -m evaluation.trust_score.rag_ab.score_ragas       # full
"""
from __future__ import annotations

import json

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.metrics import AnswerCorrectness

from dc_available.config.llm_config import get_langchain_llm
from evaluation.trust_score.rag_ab.config import AB_ARM_FILES, AB_SCORES_FILE

ARMS = ["off", "prior", "full"]
# Factual-only: statement-presence F1 (answer vs gold). No embeddings needed.
_METRIC = AnswerCorrectness(weights=[1.0, 0.0])


class _NoEmbeddings(Embeddings):
    """No-op embeddings. answer_correctness with weights [1.0, 0.0] never calls
    these, but ragas.evaluate() eagerly builds a default OpenAI embeddings when
    none is passed (which needs OPENAI_API_KEY). Passing this skips that entirely
    so the run uses only the az-login LLM — no extra API/key required."""

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


def _load_done() -> set:
    done = set()
    if AB_SCORES_FILE.is_file():
        for line in AB_SCORES_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["question_id"], r["arm"]))
    return done


def _rows_for_arm(arm: str) -> list[dict]:
    path = AB_ARM_FILES[arm]
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def run(limit: int | None = None) -> None:
    llm = get_langchain_llm("gpt-4o")
    done = _load_done()

    for arm in ARMS:
        rows = _rows_for_arm(arm)
        if limit:
            rows = rows[:limit]
        todo = [r for r in rows if (r["question_id"], arm) not in done]
        if not todo:
            print(f"arm '{arm}': nothing to score ({len(rows)} rows already done)")
            continue

        print(f"\n=== arm '{arm}': scoring {len(todo)} answers with RAGAS ===", flush=True)
        ds = Dataset.from_dict({
            "question": [r["question"] for r in todo],
            "answer": [r["answer"] for r in todo],
            "contexts": [r.get("contexts", []) for r in todo],
            "ground_truth": [r["ground_truth"] for r in todo],
        })
        result = evaluate(ds, metrics=[_METRIC], llm=llm,
                          embeddings=_NoEmbeddings(), raise_exceptions=False)
        scores = result.to_pandas()["answer_correctness"].tolist()

        with AB_SCORES_FILE.open("a", encoding="utf-8") as fh:
            for r, s in zip(todo, scores):
                val = None if s is None or (isinstance(s, float) and s != s) else round(float(s), 4)
                fh.write(json.dumps({
                    "question_id": r["question_id"], "arm": arm,
                    "lever": r.get("lever"), "answer_correctness": val,
                }, ensure_ascii=False) + "\n")
        ok = [s for s in scores if s is not None and s == s]
        mean = f"{sum(ok) / len(ok):.3f}" if ok else "n/a"
        print(f"  arm '{arm}': {len(ok)}/{len(todo)} scored, mean={mean}", flush=True)

    print(f"\n-> {AB_SCORES_FILE.name}")


if __name__ == "__main__":
    import sys
    run(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)

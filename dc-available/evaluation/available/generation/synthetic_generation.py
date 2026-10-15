"""Synthetic generation + RAGAS — hybrid leg only.

Step 1 (generate): answer each hybrid question from its stored retrieved_contexts
                    (no re-retrieval) using the Azure LLM judge.
Step 2 (ragas):     score the answers with 4 RAGAS metrics.

Reads   evaluation/results/synthetic/retrieval.json   (hybrid rows)
Writes  evaluation/results/synthetic/ :
  generation.json     hybrid rows + generated_answer
  ragas_scores.json   per-question metric values (+ .csv)
  ragas_summary.json  mean per metric

Safe to re-run: rows already answered in generation.json are reused.
Run from the project root (dc-available):
  poetry run python -m evaluation.available.generation.synthetic_generation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path("evaluation/results/synthetic")
RETRIEVAL_JSON = OUT_DIR / "retrieval.json"
GENERATION_JSON = OUT_DIR / "generation.json"
JUDGE_MODEL = "gpt-4o-mini"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

_SYSTEM = (
    "You are a technical assistant. Answer the question based solely on the "
    "provided context. Be concise, accurate, and self-contained. "
    "Do not say 'according to the context' or similar phrases."
)


def _strip_paths() -> None:
    """Keep the local 'datasets' folder from shadowing HuggingFace's 'datasets'."""
    here = Path(__file__).resolve().parent                       # .../src/synthetic
    bad = {str(here), str(here.parents[0]), str(here.parents[1])}  # synthetic, src, evaluation
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in bad]


# --- step 1: generate answers ------------------------------------------------
def _build_llm():
    from dc_available.config.llm_config import get_langchain_llm
    return get_langchain_llm(model_name=JUDGE_MODEL)


def _generate(llm, question: str, contexts: list[str]) -> str:
    from langchain.schema import HumanMessage, SystemMessage
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    msgs = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {question}"),
    ]
    return llm.invoke(msgs).content.strip()


def generate() -> list[dict]:
    if not RETRIEVAL_JSON.exists():
        sys.exit(f"Missing {RETRIEVAL_JSON}  —  run retrieval.py first.")
    records = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))
    hybrid = [r for r in records if r.get("retrieval_type") == "hybrid"]
    print(f"{len(hybrid)} hybrid rows")

    cached: dict[str, str] = {}
    if GENERATION_JSON.exists():
        for r in json.loads(GENERATION_JSON.read_text(encoding="utf-8")):
            if r.get("generated_answer"):
                cached[r["question_id"]] = r["generated_answer"]
        print(f"{len(cached)} already generated (will skip)")

    llm = _build_llm()
    results = []
    for i, row in enumerate(hybrid, 1):
        qid = row["question_id"]
        if qid in cached:
            results.append({**row, "generated_answer": cached[qid]})
            continue
        contexts = [c for c in (row.get("retrieved_contexts") or []) if c]
        if not contexts:
            print(f"  [{i}/{len(hybrid)}] {qid}: no contexts — skipped")
            results.append({**row, "generated_answer": ""})
            continue
        try:
            ans = _generate(llm, row["question"], contexts)
            print(f"  [{i}/{len(hybrid)}] {qid}: OK ({len(ans)} chars)")
        except Exception as exc:
            print(f"  [{i}/{len(hybrid)}] {qid}: ERROR {exc}")
            ans = ""
        results.append({**row, "generated_answer": ans})

    GENERATION_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {GENERATION_JSON}")
    return results


# --- step 2: RAGAS scoring ---------------------------------------------------
def _build_ragas_llm_emb():
    from dc_available.config.llm_config import get_langchain_llm
    from dc_available.RAG.retrieval_helpers import get_embeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    llm = LangchainLLMWrapper(get_langchain_llm(model_name=JUDGE_MODEL))
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    return llm, emb


def _build_hf_dataset(rows: list[dict]):
    from datasets import Dataset as HFDataset
    items, idx = [], []
    for i, r in enumerate(rows):
        ans = (r.get("generated_answer") or "").strip()
        ctxs = [c for c in (r.get("retrieved_contexts") or []) if c]
        if not ans or not ctxs:
            continue
        gt = (r.get("golden_answer") or "").strip()
        items.append({
            "question": r["question"],
            "answer": ans,
            "contexts": ctxs,
            "ground_truths": [gt] if gt else [""],
        })
        idx.append(i)
    return HFDataset.from_list(items), idx


def ragas_score(rows: list[dict]) -> None:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )

    ragas_llm, ragas_emb = _build_ragas_llm_emb()
    dataset, indices = _build_hf_dataset(rows)
    print(f"{len(dataset)} rows to score (of {len(rows)} hybrid rows)")
    if len(dataset) == 0:
        print("nothing to score — skipping")
        return

    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    df = result.to_pandas()
    qids = [rows[i]["question_id"] for i in indices[: len(df)]]
    df.insert(0, "question_id", qids)
    df.to_csv(OUT_DIR / "ragas_scores.csv", index=False)
    df.to_json(OUT_DIR / "ragas_scores.json", orient="records", indent=2, force_ascii=False)

    summary = {
        m: round(float(df[m].mean(skipna=True)), 4)
        for m in METRIC_NAMES if m in df.columns
    }
    (OUT_DIR / "ragas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RAGAS summary:", summary)
    print(f"→ {OUT_DIR}/ragas_scores.json, ragas_scores.csv, ragas_summary.json")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _strip_paths()
    rows = generate()
    ragas_score(rows)


if __name__ == "__main__":
    main()

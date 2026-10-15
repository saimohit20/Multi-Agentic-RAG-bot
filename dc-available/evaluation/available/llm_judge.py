from __future__ import annotations

import os

os.environ["OPIK_TRACK_DISABLE"] = "true"
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# Silence noisy loggers that don't affect results (see reranker_ablation.py).
import logging

logging.getLogger("opik").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.ERROR)

import sys
from pathlib import Path

# Keep the project root on sys.path; drop the eval dirs so local "evaluation/
# datasets" can't shadow HuggingFace's "datasets". (Same guard as the ablation.)
_here = Path(__file__).resolve().parent
_root = _here.parents[1]
_drop = {str(_here), str(_here.parent)}
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _drop]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

from dc_available.config.llm_config import get_langchain_llm

JUDGE_MODEL = "gpt-4o"
CONFIGS = ["no_rerank", "base", "v2m3"]
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
RAGAS_COLS = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
DEFAULT_RESULTS = Path("evaluation/results/reranker_ablation/grownfield")
DEFAULT_N_LABEL = 20


# --- structured output ------------------------------------------------------
class Verdict(BaseModel):
    """One correctness judgement. `reasoning` comes first so the model thinks
    before committing to a verdict (chain-of-thought improves judging accuracy)."""

    reasoning: str = Field(
        description="Brief step-by-step check: identify the key fact(s) in the reference, "
        "then state whether each is present and correct in the generated answer."
    )
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="correct: all key facts from the reference are present and accurate. "
        "partial: core is right but some required facts are missing or slightly imprecise. "
        "incorrect: the key fact is missing, wrong, or contradicts the reference."
    )
    rationale: str = Field(description="One sentence naming the specific fact(s) the verdict is based on.")


_SYSTEM = (
    "You are a strict but fair evaluator of question-answering systems.\n"
    "You compare a GENERATED answer against a REFERENCE (gold) answer and decide\n"
    "how correct the generated answer is.\n\n"
    "Rules:\n"
    "- Judge only CORRECTNESS against the reference.\n"
    "- The generated answer is often longer than the reference. Extra detail,\n"
    "  explanation, or context is NOT a penalty, as long as it is not wrong and\n"
    "  does not contradict the reference.\n"
    "- Focus on whether the key fact(s) in the reference are present and correct\n"
    "  in the generated answer.\n"
    "- Do not reward or punish style, length, tone, or formatting.\n"
    "- IMPORTANT: when the reference is a specific identifier, transaction code, or\n"
    "  exact value (e.g. 'VA01', 'MSR_TRC_C', 'ZT'), the generated answer must\n"
    "  contain that EXACT identifier to be correct. A different/wrong identifier is\n"
    "  'incorrect'; a correct process that omits the required identifier is 'partial'.\n\n"
    "Verdicts:\n"
    "- \"correct\"   : all key facts from the reference are present and accurate; no contradictions.\n"
    "- \"partial\"   : the core is right but some required facts are missing or slightly imprecise\n"
    "                (e.g. lists 3 of 4 required items, or right process but missing the code).\n"
    "- \"incorrect\" : the key fact is missing, wrong, or contradicts the reference.\n\n"
    "First reason step by step, then give the verdict and a one-sentence rationale."
)


def _user(question: str, golden: str, generated: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE (gold) ANSWER:\n{golden}\n\n"
        f"GENERATED ANSWER:\n{generated}\n\n"
        "Return your verdict."
    )


# --- judging ----------------------------------------------------------------
def judge_record(judge, record: dict) -> dict:
    """Score one record; returns judge fields (does not mutate input)."""
    q = str(record.get("question", ""))
    gold = str(record.get("golden_answer", ""))
    gen = str(record.get("generated_answer", ""))
    if not gen or not gold:
        return {"judge_verdict": None, "judge_score": None,
                "judge_reasoning": "", "judge_rationale": "skipped: empty generated or gold answer"}
    try:
        result = judge.invoke([("system", _SYSTEM), ("human", _user(q, gold, gen))])
        # with_structured_output may return a Verdict model or a plain dict,
        # depending on the langchain/Azure version — handle both.
        if isinstance(result, dict):
            verdict = result.get("verdict")
            rationale = str(result.get("rationale", "")).strip()
            reasoning = str(result.get("reasoning", "")).strip()
        else:
            verdict = result.verdict
            rationale = result.rationale.strip()
            reasoning = result.reasoning.strip()
        return {
            "judge_verdict": verdict,
            "judge_score": SCORE.get(verdict),
            "judge_reasoning": reasoning,
            "judge_rationale": rationale,
        }
    except Exception as exc:
        return {"judge_verdict": None, "judge_score": None,
                "judge_reasoning": "", "judge_rationale": f"error: {exc}"}


def judge_config(judge, records: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(records, 1):
        fields = judge_record(judge, r)
        out.append({**r, **fields})
        note = "" if fields["judge_verdict"] is not None else f"  <- {fields['judge_rationale']}"
        print(f"  [{i}/{len(records)}] {r.get('question_id')}: {fields['judge_verdict']}{note}")
    return out


# --- RAGAS 4 metrics --------------------------------------------------------
def score_ragas(records: list[dict]) -> None:
    """Add RAGAS faithfulness / answer_relevancy / context_precision /
    answer_correctness per record (in place). Best-effort: on any failure the
    ragas_* fields stay None so the LLM-judge results are never lost."""
    for r in records:
        r.update({f"ragas_{c}": None for c in RAGAS_COLS})
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, answer_correctness,
        )
        from dc_available.config.data_config import DEFAULT_EMBEDDING_MODEL
        from dc_available.RAG.retrieval_helpers import get_embeddings
    except Exception as exc:
        print(f"  [ragas] import failed, skipping RAGAS: {exc}")
        return

    valid = [r for r in records
             if r.get("generated_answer") and r.get("retrieved_contexts") and r.get("golden_answer")]
    if not valid:
        print("  [ragas] no valid rows to score.")
        return
    data = {
        "question": [r["question"] for r in valid],
        "answer": [r["generated_answer"] for r in valid],
        "contexts": [r["retrieved_contexts"] for r in valid],
        "ground_truth": [r["golden_answer"] for r in valid],
    }
    try:
        llm = get_langchain_llm(model_name="gpt-4o")
        emb = get_embeddings(DEFAULT_EMBEDDING_MODEL)
        res = evaluate(
            Dataset.from_dict(data),
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
            llm=llm, embeddings=emb,
        )
        df = res.to_pandas()
        for i, r in enumerate(valid):
            for c in RAGAS_COLS:
                v = df.iloc[i].get(c)
                r[f"ragas_{c}"] = round(float(v), 4) if v is not None and v == v else None
        print(f"  [ragas] scored {len(valid)} rows.")
    except Exception as exc:
        print(f"  [ragas] evaluate failed, leaving RAGAS fields None: {exc}")


# --- summary ----------------------------------------------------------------
def summarize(name: str, records: list[dict]) -> dict:
    scored = [r for r in records if r.get("judge_score") is not None]
    n = len(scored) or 1
    counts = {v: sum(1 for r in scored if r["judge_verdict"] == v)
              for v in ("correct", "partial", "incorrect")}
    row = {
        "config": name,
        "n_scored": len(scored),
        "judge_correctness": round(sum(r["judge_score"] for r in scored) / n, 4),
        **{f"n_{v}": counts[v] for v in counts},
    }
    for c in RAGAS_COLS:
        vals = [r.get(f"ragas_{c}") for r in records if r.get(f"ragas_{c}") is not None]
        row[f"ragas_{c}"] = round(sum(vals) / len(vals), 4) if vals else None
    return row


# --- manual-validation sample ----------------------------------------------
def write_label_sheet(all_judged: dict[str, list[dict]], out: Path, n: int) -> None:
    """Even sample across configs with an empty human_label column to fill in."""
    rows = []
    per_cfg = max(1, n // len(all_judged))
    for cfg, records in all_judged.items():
        for r in records[:per_cfg]:
            rows.append({
                "question_id": r.get("question_id"),
                "config": cfg,
                "question": r.get("question"),
                "golden_answer": r.get("golden_answer"),
                "generated_answer": r.get("generated_answer"),
                "judge_verdict": r.get("judge_verdict"),
                "judge_reasoning": r.get("judge_reasoning"),
                "judge_rationale": r.get("judge_rationale"),
                "human_label": "",      # <- you fill: correct / partial / incorrect
            })
    pd.DataFrame(rows).to_csv(out / "to_label.csv", index=False, encoding="utf-8")
    print(f"  wrote {len(rows)} rows to {out / 'to_label.csv'} (fill the human_label column)")


# --- orchestrator -----------------------------------------------------------
def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    n_label = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_LABEL

    judge = get_langchain_llm(model_name=JUDGE_MODEL, temperature=0).with_structured_output(Verdict)

    all_judged: dict[str, list[dict]] = {}
    summary_rows = []
    for cfg in CONFIGS:
        path = results_dir / f"{cfg}.json"
        if not path.exists():
            print(f"[skip] {path} not found — run reranker_ablation first.")
            continue
        print(f"\n=== judging {cfg} ===")
        records = json.loads(path.read_text(encoding="utf-8"))
        judged = judge_config(judge, records)
        print(f"  [ragas] scoring 4 metrics for {cfg} ...")
        score_ragas(judged)
        all_judged[cfg] = judged
        (results_dir / f"{cfg}_judged.json").write_text(
            json.dumps(judged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_rows.append(summarize(cfg, judged))

    if not summary_rows:
        print("Nothing judged.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "judge_summary.csv", index=False)
    write_label_sheet(all_judged, results_dir, n_label)

    print(f"\n=== LLM-judge correctness ({JUDGE_MODEL}) + RAGAS 4 metrics (all questions) ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote *_judged.json + judge_summary.csv + to_label.csv to {results_dir}")


if __name__ == "__main__":
    main()

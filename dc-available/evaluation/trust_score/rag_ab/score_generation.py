"""Step 7 — score the generated answers (the PRIMARY generation graders).

Two graders per (question, arm), written to ab_gen.jsonl:
  - exact      : ground-truth match — is `correct_value` present in the answer
                 (case-insensitive)? Deterministic, no LLM. Also `twin` = did the
                 answer follow the twin's wrong_value. This is the headline metric,
                 valid because we authored the gold answers.
  - llm_judge  : an own gpt-4o judge built for SHORT SAP answers (a code / app name
                 must match exactly, MM01 != MM04; extra explanation is ignored).
                 Independent validation that confirms `exact` and isolates RAGAS
                 (which mis-scores short answers) as the outlier.

RAGAS answer_correctness lives in score_ragas.py / ab_scores.jsonl and is kept only
as the "off-the-shelf metric fails on short answers" exhibit (see analyze_ab).

Resumable per (question_id, arm). Pass an integer to limit questions (dry-run).

RUN (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.score_generation 6   # dry-run: first 6 per arm
    python -m evaluation.trust_score.rag_ab.score_generation      # full
"""
from __future__ import annotations

import json

from dc_available.config.llm_config import get_openai_client
from evaluation.trust_score.rag_ab.config import AB_ARM_FILES, AB_GEN_FILE, AB_SCORES_FILE

ARMS = ["off", "prior", "full"]
_MODEL = "gpt-4o"

_JUDGE_SYSTEM = """\
You grade a question-answering result for SAP S/4HANA training content.
You are given a QUESTION, the correct GOLD answer, and a model ANSWER.

Decide whether the ANSWER conveys the SAME underlying fact as GOLD:
- Ignore differences in case, spacing, hyphens, underscores, punctuation, surrounding
  words, language, and formatting. "QP01", "qp-01", and "QP 01" are the SAME value.
- But the identifier VALUE itself must be the same: a different transaction code, app
  name, or number is WRONG (e.g. MM01 vs MM04, QI02 vs QI05, "Manage Product Master
  Data" vs "Manage Material Master Data" are all WRONG).
- If the ANSWER states it cannot find the information, it is not correct.

Return ONLY this JSON: {"correct": true or false}"""


def _judge(client, question: str, gold: str, answer: str) -> bool | None:
    """gpt-4o verdict; None on parse/call failure (kept out of the file to retry later)."""
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",
                 "content": f"QUESTION: {question}\nGOLD: {gold}\nANSWER: {answer}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return bool(data.get("correct"))
    except Exception as exc:                       # noqa: BLE001
        print(f"    judge failed: {exc}", flush=True)
        return None


def _load_done() -> set:
    done = set()
    if AB_GEN_FILE.is_file():
        for line in AB_GEN_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["question_id"], r["arm"]))
    return done


def _load_ragas() -> dict[tuple, float]:
    """(question_id, arm) -> RAGAS answer_correctness, merged into the unified file."""
    out = {}
    if AB_SCORES_FILE.is_file():
        for line in AB_SCORES_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            v = r.get("answer_correctness")
            if v is not None:
                out[(r["question_id"], r["arm"])] = float(v)
    return out


def run(limit: int | None = None) -> None:
    client = get_openai_client()
    done = _load_done()
    ragas = _load_ragas()
    if not ragas:
        print("note: no RAGAS scores found (ab_scores.jsonl) — 'answer_correctness'/'ragas' "
              "will be null. Run score_ragas.py first to fill them.")

    for arm in ARMS:
        path = AB_ARM_FILES[arm]
        if not path.is_file():
            print(f"arm '{arm}': no file ({path.name}) — skipping")
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if limit:
            rows = rows[:limit]
        todo = [r for r in rows if (r["question_id"], arm) not in done]
        if not todo:
            print(f"arm '{arm}': nothing to score ({len(rows)} already done)")
            continue

        print(f"\n=== arm '{arm}': grading {len(todo)} answers ===", flush=True)
        for i, r in enumerate(todo, 1):
            answer = (r.get("answer") or "")
            cv, wv = str(r.get("correct_value", "")), str(r.get("wrong_value", ""))
            exact = bool(cv) and cv.lower() in answer.lower()
            twin = bool(wv) and wv.lower() in answer.lower()
            verdict = _judge(client, r["question"], r.get("ground_truth", ""), answer)
            rg = ragas.get((r["question_id"], arm))

            out = {
                "question_id": r["question_id"], "arm": arm, "lever": r.get("lever"),
                "exact": exact, "twin": twin, "llm_judge": verdict,
                "answer_correctness": rg,                       # raw RAGAS (float) or null
                "ragas": (rg >= 0.5) if rg is not None else None,   # RAGAS as bool verdict
            }
            with AB_GEN_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  {arm}: {i}/{len(todo)}", flush=True)

    print(f"\n-> {AB_GEN_FILE.name}")


if __name__ == "__main__":
    import sys
    run(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)

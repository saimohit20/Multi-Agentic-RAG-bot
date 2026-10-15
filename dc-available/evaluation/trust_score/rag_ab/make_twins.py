"""Step 2 — build the twins, the ~200-question dataset, and the test trust table.

For each of the 20 good docs:
  1. gpt-5.2 picks ~10 concrete, checkable facts from the summary and, per fact,
     gives correct_value / wrong_value / question / gold_answer.
  2. We build the TWIN text by replacing each correct_value with its wrong_value
     in the good summary (deterministic, word-boundary — so exactly the flipped
     facts change, nothing else).
  3. We compute a trust pair with the real `score_prior`: the good doc keeps its
     genuine prior; the twin is demoted purely via its lever (freshness → backdate
     4y, authority → unknown), so the trust gap is caused by the lever alone.

Outputs (all under rag_ab/):
  - twins/<doc_id>_twin.txt      the degraded summary text (indexed as the twin)
  - dataset.jsonl                ~200 rows (see columns below)
  - test_trust.jsonl             40 rows: g01..g20 (good, high) + *_twin (low)

Resumable: docs whose twin file already exists are reused (no re-billing).

RUN (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.make_twins
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dc_available.config.llm_config import get_openai_client
from dc_available.trust_score.components.score_content import _read_text
from dc_available.trust_score.components.score_prior import (
    AUTHORITY,
    WEIGHTS,
    authority_score,
    format_score,
    freshness_score,
)
from evaluation.trust_score.rag_ab.config import (
    CORRUPT_MODEL,
    DATASET_FILE,
    DOCS_FILE,
    FACTS_PER_TWIN,
    TEST_TRUST_FILE,
    TWINS_DIR,
)

_BACKDATE_YEARS = 4

_FACT_PROMPT = f"""\
You are given a SUMMARY of an SAP S/4HANA training document.
Pick exactly {FACTS_PER_TWIN} DISTINCT, concrete, checkable facts from it. Each fact
must be a specific token that appears VERBATIM in the summary — a transaction code
(e.g. VA01), program/report name, table, field value, ID, or a specific number.

For each fact return:
- "fact": a short description of what the value is
- "correct_value": the EXACT token as written in the summary (copy it verbatim)
- "wrong_value": a DIFFERENT but plausible value of the SAME type/format that does
  NOT appear anywhere in the summary (e.g. change VA01 -> VA07, MM03 -> MM09)
- "question": a natural question a trainee would ask whose answer is exactly correct_value
- "gold_answer": correct_value

Rules: choose tokens with a UNIQUE, unambiguous correct_value (avoid single letters
or bare small numbers that occur many times). Do not reuse a correct_value twice.
Return ONLY JSON: {{"facts": [ {{...}}, ... ]}}"""


def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        return json.loads(t[i:j + 1])


def _gen_facts(client, summary_text: str, retries: int = 3) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=CORRUPT_MODEL,
                messages=[
                    {"role": "system", "content": _FACT_PROMPT},
                    {"role": "user", "content": f"SUMMARY:\n{summary_text}"},
                ],
            )
            data = _parse_json(resp.choices[0].message.content or "")
            return list(data.get("facts", []))
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"      ! {type(exc).__name__}: {exc} (retry {attempt}/{retries})", flush=True)
            time.sleep(3 * attempt)


def _valid_facts(facts: list[dict], summary: str) -> list[dict]:
    """Keep only facts whose correct_value is verbatim in the summary, whose
    wrong_value differs and is absent, deduped by correct_value."""
    out, seen = [], set()
    for f in facts:
        cv = str(f.get("correct_value", "")).strip()
        wv = str(f.get("wrong_value", "")).strip()
        if not cv or not wv or cv == wv or cv in seen:
            continue
        if cv not in summary or wv in summary:
            continue
        if not (f.get("question") and f.get("gold_answer")):
            continue
        seen.add(cv)
        out.append(f)
    return out


def _swap(text: str, correct: str, wrong: str) -> str:
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(correct) + r"(?![A-Za-z0-9])")
    return pat.sub(lambda _m: wrong, text)


def _backdate(modified_iso: str, years: int) -> str:
    if not modified_iso:
        return modified_iso
    try:
        dt = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
    except ValueError:
        return modified_iso
    try:
        return dt.replace(year=dt.year - years).isoformat()
    except ValueError:
        return dt.replace(year=dt.year - years, day=28).isoformat()


def _trust_pair(d: dict, now: datetime) -> tuple[float, float]:
    """Good prior and twin prior via the real score_prior; only the lever differs."""
    content = float(d.get("content_score") or 1.0)
    fmt = format_score(d.get("file_type", ""))
    auth = authority_score(d.get("created_by", ""), d.get("modified_by", ""))
    fresh = freshness_score(d.get("modified", ""), now)

    def prior(a: float, fr: float) -> float:
        return round(WEIGHTS["content"] * content + WEIGHTS["authority"] * a
                     + WEIGHTS["format"] * fmt + WEIGHTS["freshness"] * fr, 4)

    good = prior(auth, fresh)
    if d["lever"] == "freshness":
        twin = prior(auth, freshness_score(_backdate(d.get("modified", ""), _BACKDATE_YEARS), now))
    else:  # authority
        twin = prior(AUTHORITY["unknown"], fresh)
    return good, twin


def run(resume: bool = True) -> None:
    docs = [json.loads(line) for line in DOCS_FILE.open(encoding="utf-8")]
    TWINS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    existing: dict[str, list] = {}
    if resume and DATASET_FILE.is_file():
        for line in DATASET_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            existing.setdefault(r["good_doc_id"], []).append(r)

    client = None
    dataset, trust = [], []
    print(f"twinning {len(docs)} docs with '{CORRUPT_MODEL}' ...", flush=True)

    for n, d in enumerate(docs, 1):
        did, lever = d["doc_id"], d["lever"]
        twin_file = TWINS_DIR / f"{did}_twin.txt"
        good_trust, twin_trust = _trust_pair(d, now)
        trust.append({"match_key": did, "final_trust": good_trust})
        trust.append({"match_key": f"{did}_twin", "final_trust": twin_trust})

        if resume and twin_file.is_file() and existing.get(did):
            dataset.extend(existing[did])
            print(f"[{n}/{len(docs)}] {did}: reuse ({len(existing[did])} q)", flush=True)
            continue

        good_text = _read_text(Path(d["summary_path"]))
        if client is None:
            client = get_openai_client()
        print(f"[{n}/{len(docs)}] {did} ({lever}): calling {CORRUPT_MODEL} ...", flush=True)
        facts = _valid_facts(_gen_facts(client, good_text), good_text)

        twin_text = good_text
        for f in facts:
            twin_text = _swap(twin_text, f["correct_value"], f["wrong_value"])
        twin_file.write_text(twin_text, encoding="utf-8")

        for i, f in enumerate(facts, 1):
            dataset.append({
                "question_id": f"{did}_q{i:02d}",
                "good_doc_id": did, "twin_doc_id": f"{did}_twin", "lever": lever,
                "fact": f.get("fact", ""),
                "correct_value": f["correct_value"], "wrong_value": f["wrong_value"],
                "question": f["question"], "gold_answer": f.get("gold_answer", f["correct_value"]),
            })
        print(f"        -> {len(facts)} facts kept", flush=True)

    with DATASET_FILE.open("w", encoding="utf-8") as fh:
        for r in dataset:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TEST_TRUST_FILE.open("w", encoding="utf-8") as fh:
        for r in trust:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\ndataset : {len(dataset)} questions -> {DATASET_FILE.name}")
    print(f"trust   : {len(trust)} rows -> {TEST_TRUST_FILE.name}")
    gt = next((r['final_trust'] for r in trust if not r['match_key'].endswith('_twin')), None)
    print(f"example trust: good={gt}  "
          f"twin(freshness/authority) demoted via lever")


if __name__ == "__main__":
    run()

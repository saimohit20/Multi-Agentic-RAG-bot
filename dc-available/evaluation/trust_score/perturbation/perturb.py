"""Step 2 — build the 240-row manifest and the 120 perturbed summaries.

For each of the 40 docs we emit 6 manifest rows:
    - 1 original                       (input: "original")
    - 3 content perturbations          (input: {"summary_file": ...})  <- gpt-5.2
    - 2 metadata perturbations         (input: {"override": {...}})    <- no LLM

Content perturbations corrupt the document SUMMARY (the scorer's actual input)
with gpt-5.2 and save it under perturbation/summaries/. Metadata perturbations
carry an inline field override and touch no file. Resumable: existing summary
files are not regenerated (avoids re-billing).

Input : perturbation/sample.jsonl
Output: perturbation/perturbations.jsonl  (+ perturbation/summaries/*.txt)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.perturbation.perturb
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from dc_available.config.llm_config import get_openai_client
from dc_available.trust_score.components.score_content import _read_text
from evaluation.trust_score.perturbation.config import (
    CONTENT_TYPES,
    CORRUPT_MODEL,
    FRESHNESS_BACKDATE_YEARS,
    MANIFEST_FILE,
    SAMPLE_FILE,
    SUMMARIES_DIR,
)
from evaluation.trust_score.perturbation.prompts import CORRUPT_PROMPTS


def _corrupt(client, system_prompt: str, summary_text: str, retries: int = 3) -> str:
    """One corruption call: degrade the summary. Temperature left at default for
    broad model compatibility (gpt-5.x may reject temperature=0). Retries a few
    times on transient connection drops so one blip doesn't kill the run."""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=CORRUPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"SUMMARY:\n{summary_text}"},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 3 * attempt
            print(f"      ! {type(exc).__name__}: {exc} "
                  f"(attempt {attempt}/{retries}, retry in {wait}s)", flush=True)
            time.sleep(wait)


def _backdate(modified_iso: str, years: int) -> str:
    if not modified_iso:
        return modified_iso
    try:
        dt = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
    except ValueError:
        return modified_iso
    try:
        return dt.replace(year=dt.year - years).isoformat()
    except ValueError:  # Feb 29 etc.
        return dt.replace(year=dt.year - years, day=28).isoformat()


def run(resume: bool = True) -> None:
    docs = [json.loads(line) for line in SAMPLE_FILE.open(encoding="utf-8")]
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    client = None
    manifest: list[dict] = []
    made = skipped = 0

    print(f"corrupting {len(docs)} docs with '{CORRUPT_MODEL}' ...", flush=True)
    for n_i, d in enumerate(docs, 1):
        did, mk = d["doc_id"], d["match_key"]
        print(f"[{n_i}/{len(docs)}] {did}  {mk[:45]}", flush=True)

        manifest.append({"id": did, "base": did, "match_key": mk,
                         "perturbation": "none", "input": "original"})

        orig_summary = _read_text(Path(d["summary_path"]))
        for t in CONTENT_TYPES:
            out = SUMMARIES_DIR / f"{did}__{t}.txt"
            if resume and out.is_file():
                print(f"    - {t}: reuse", flush=True)
                skipped += 1
            else:
                if client is None:
                    client = get_openai_client()
                print(f"    - {t}: calling {CORRUPT_MODEL} ...", flush=True)
                out.write_text(_corrupt(client, CORRUPT_PROMPTS[t], orig_summary),
                               encoding="utf-8")
                made += 1
            manifest.append({"id": f"{did}__{t}", "base": did, "match_key": mk,
                             "perturbation": t,
                             "input": {"summary_file": f"summaries/{out.name}"}})

        manifest.append({"id": f"{did}__authority", "base": did, "match_key": mk,
                         "perturbation": "authority",
                         "input": {"override": {"created_by": "", "modified_by": ""}}})
        manifest.append({"id": f"{did}__freshness", "base": did, "match_key": mk,
                         "perturbation": "freshness",
                         "input": {"override": {
                             "modified": _backdate(d.get("modified", ""), FRESHNESS_BACKDATE_YEARS)}}})

    with MANIFEST_FILE.open("w", encoding="utf-8") as fh:
        for r in manifest:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"manifest       : {len(manifest)} rows -> {MANIFEST_FILE.name}")
    print(f"summaries      : {made} generated, {skipped} reused ({CORRUPT_MODEL})")


if __name__ == "__main__":
    run()

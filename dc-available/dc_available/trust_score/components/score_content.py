"""Stage B (content pillar) — score each document's CONTENT QUALITY.

Reads the aligned metadata, and for every document produces a 0-1 content_score:

    * .mp4 videos            -> 1.0 (full), no LLM call        (decided with user)
    * text docs (has summary)-> LLM rates the __summary.txt on 4 domain criteria

We score the LLM-generated __summary.txt (JSON or prose), NOT raw text, because
that is the cleaned representation the pipeline produced. The rubric is tuned to
SAP S/4HANA training material (transaction codes, order/item types, program
names, tables, GUI steps) — see SYSTEM_PROMPT.

Output: content_scores.jsonl (one row per document, keyed by match_key).
Resumable: already-scored keys are kept, so a re-run only fills the gaps and
does not re-bill the LLM.

RUN (from the dc-available directory):
    python -m dc_available.trust_score.prior_score
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from dc_available.config.llm_config import get_openai_client

_MODEL = "gpt-4o"
_SUB_KEYS = ["technical_depth", "process_completeness", "training_usefulness", "focus"]

_DATA = Path(__file__).resolve().parent.parent / "data"
_TRUST_DIR = _DATA / "Artifacts" / "resources" / "trust" / "grownfield"
_METADATA = _TRUST_DIR / "metadata_aligned.jsonl"
_OUT = _TRUST_DIR / "content_scores.jsonl"
_INGESTION = _DATA / "Artifacts" / "resources" / "ingestion" / "GROWNFIELD"

SYSTEM_PROMPT = """\
You are assessing the CONTENT QUALITY of an SAP S/4HANA training document.
You are given an LLM-generated SUMMARY of it (JSON or prose). Judge how useful
the underlying document is as TRAINING material, based on what the summary
reveals. Do NOT judge the summary's writing/grammar, and do NOT judge whether
the facts are correct — only how substantive and usable the training content is.

Score each 0.0-1.0:
- technical_depth       : concrete SAP specifics (transaction codes, order/item
                          types, program names, tables, fields, config) vs. vague generalities
- process_completeness  : a complete followable procedure (prerequisites -> steps
                          -> outcome) vs. an isolated fragment or single screen
- training_usefulness   : a user could actually perform the task from it
                          (clear actions, what to enter/click, and why)
- focus                 : one coherent task/topic vs. a jumble or mostly filler
                          describing slide visuals

Penalize summaries that mainly describe visual/slide appearance (e.g. "a slide
with a blue background") with little real procedure — that signals a thin,
image-only source.

Return ONLY this JSON:
{
  "technical_depth": <0-1>,
  "process_completeness": <0-1>,
  "training_usefulness": <0-1>,
  "focus": <0-1>,
  "reason": "<one short sentence citing specifics from the summary>"
}"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _summary_index() -> dict[str, Path]:
    """stem (lower) -> __summary.txt path, matching the metadata match_key."""
    idx: dict[str, Path] = {}
    for f in _INGESTION.rglob("*__summary.txt"):
        base = f.name.replace("__summary.txt", "")
        stem = base.rsplit(".", 1)[0].strip().lower() if "." in base else base.strip().lower()
        idx.setdefault(stem, f)
    return idx


def _read_text(p: Path) -> str:
    """Some summaries are Windows-1252 (en-dash 0x96), not UTF-8 — try both."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="replace")


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _parse_json(text: str) -> dict:
    """Parse the model's reply, tolerating ```json fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    return json.loads(t.strip())


def _score_summary(client, summary_text: str) -> dict:
    resp = client.chat.completions.create(
        model=_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT SUMMARY:\n{summary_text}"},
        ],
    )
    data = _parse_json(resp.choices[0].message.content)
    subs = {k: _clamp(data.get(k)) for k in _SUB_KEYS}
    subs["content_score"] = round(mean(subs.values()), 4)
    subs["reason"] = str(data.get("reason", ""))[:300]
    subs["source"] = "llm"
    return subs


def _video_row() -> dict:
    row = {k: 1.0 for k in _SUB_KEYS}
    row["content_score"] = 1.0
    row["reason"] = "video/transcript — full content score by policy"
    row["source"] = "video_default"
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run() -> None:
    metadata = [json.loads(line) for line in _METADATA.open(encoding="utf-8")]

    # Resume: keep good rows, but drop 'error'/'no_summary' so they get retried
    # this run (dict keyed by match_key also collapses duplicates).
    done: dict[str, dict] = {}
    if _OUT.is_file():
        for line in _OUT.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("source") not in ("error", "no_summary"):
                done[r["match_key"]] = r
        print(f"resuming — {len(done)} already scored")

    summaries = _summary_index()
    client = None
    counts = {"video": 0, "llm": 0, "no_summary": 0, "error": 0, "skipped": 0}

    for row in metadata:
        key = row["match_key"]
        if key in done:
            counts["skipped"] += 1
            continue

        if row.get("file_type") in ("mp4", "srt"):   # video or its transcript
            out = _video_row()
            counts["video"] += 1
        else:
            sf = summaries.get(key)
            if sf is None:
                out = {**{k: 0.0 for k in _SUB_KEYS}, "content_score": None,
                       "reason": "no summary found", "source": "no_summary"}
                counts["no_summary"] += 1
            else:
                if client is None:
                    client = get_openai_client()
                try:
                    out = _score_summary(client, _read_text(sf))
                    counts["llm"] += 1
                except Exception as exc:  # keep going; a re-run retries this key
                    out = {**{k: 0.0 for k in _SUB_KEYS}, "content_score": None,
                           "reason": f"error: {exc}", "source": "error"}
                    counts["error"] += 1

        out["match_key"] = key
        done[key] = out
        # Append-as-we-go so a crash mid-run loses nothing.
        with _OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        if counts["llm"] % 20 == 0 and counts["llm"]:
            print(f"  ...{counts['llm']} LLM-scored")

    # Rewrite a clean, de-duplicated file (one row per match_key).
    with _OUT.open("w", encoding="utf-8") as fh:
        for r in done.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done -> {_OUT.name}")
    print(f"  llm={counts['llm']}  video={counts['video']}  "
          f"no_summary={counts['no_summary']}  error={counts['error']}  "
          f"already_done={counts['skipped']}")
    if counts["error"]:
        print("  (errors will be retried on the next run)")


if __name__ == "__main__":
    run()

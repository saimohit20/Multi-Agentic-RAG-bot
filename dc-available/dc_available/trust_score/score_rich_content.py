"""Stage B (RICH content pillar) — score each interview's CONTENT QUALITY.

The RICH analogue of ``components/score_content.py`` (which is GROWNFIELD-only and uses an
SAP-how-to rubric interviews would fail for the wrong reasons). This is a **standalone content
judge**: it produces one 0-1 ``content_score`` per interview and nothing else. The other prior
pillars (authority / format / freshness) are independent and handled elsewhere — this module
makes no assumption about them.

We score the LLM-generated interview SUMMARY (``summary.txt`` — the clean markdown the pipeline
already produced per interview), NOT raw transcript. The rubric rewards concrete, first-hand,
insightful knowledge over generic course-overview text (see SYSTEM_PROMPT).

Input : Artifacts/rich/trust/metadata.jsonl      (rows from collect_rich_metadata)
Output: Artifacts/rich/trust/content_scores.jsonl (one row per interview, keyed by match_key)
Resumable: already-scored keys are kept, so a re-run only fills gaps and does not re-bill the LLM.

RUN (from the dc-available directory):
    python -m dc_available.trust_score.score_rich_content
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from dc_available.config import data_config
from dc_available.config.llm_config import get_openai_client

_MODEL = "gpt-4o"
_SUB_KEYS = ["specificity", "experiential_depth", "insight"]

SYSTEM_PROMPT = """\
You are assessing an EXPERT INTERVIEW as a knowledge source for a company knowledge base.
You are given an LLM-generated SUMMARY of the interview (markdown or JSON). Judge how
substantive and trustworthy the underlying interview is as FIRST-HAND professional knowledge,
based on what the summary reveals. Do NOT judge the summary's writing/grammar, and do NOT judge
whether the facts are correct — only how substantive the knowledge is.

Score each 0.0-1.0:
- specificity        : concrete, verifiable detail — named people/companies/products, numbers,
                       € figures, dates, places, tools, standards — vs. vague generalities.
- experiential_depth : first-hand lived experience — real situations, decisions and actions the
                       interviewee took and what happened — vs. abstract, generic
                       "overview / online-training" descriptions with no personal grounding.
- insight            : reasoning and judgment — cause/effect, trade-offs, lessons learned,
                       recommendations — vs. plain description of what happened.

Penalize summaries that read like a generic training-course catalog ("this comprehensive online
training covers leadership, collaboration, innovation…") with no concrete people, numbers or
situations — that signals thin, non-specific content.

Return ONLY this JSON:
{
  "specificity": <0-1>,
  "experiential_depth": <0-1>,
  "insight": <0-1>,
  "reason": "<one short sentence citing specifics from the summary>"
}"""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _trust_dir() -> Path:
    return data_config.source_artifact_paths("rich")["ingestion"].parent / "trust"


def _ingestion_root() -> Path:
    return data_config.source_artifact_paths("rich")["ingestion"]


# --------------------------------------------------------------------------- #
# Helpers (mirrors score_content.py)
# --------------------------------------------------------------------------- #
def _summary_for(row: dict) -> Path | None:
    """Locate an interview's summary: the clean ``summary.txt`` in its folder (present for
    every interview), else any ``*__summary.txt`` fallback. The folder comes from the
    metadata ``source_rel`` so it joins reliably by interview, not by filename."""
    folder = _ingestion_root() / Path(row.get("source_rel", "")).parent
    cand = folder / "summary.txt"
    if cand.is_file():
        return cand
    hits = sorted(folder.glob("*__summary.txt"))
    return hits[0] if hits else None


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
            {"role": "user", "content": f"INTERVIEW SUMMARY:\n{summary_text}"},
        ],
    )
    data = _parse_json(resp.choices[0].message.content)
    subs = {k: _clamp(data.get(k)) for k in _SUB_KEYS}
    subs["content_score"] = round(mean(subs.values()), 4)
    subs["reason"] = str(data.get("reason", ""))[:300]
    subs["source"] = "llm"
    return subs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run() -> None:
    trust_dir = _trust_dir()
    meta_file = trust_dir / "metadata.jsonl"
    out_file = trust_dir / "content_scores.jsonl"
    metadata = [json.loads(line) for line in meta_file.open(encoding="utf-8")]

    # Resume: keep good rows, drop 'error'/'no_summary' so they are retried this run.
    done: dict[str, dict] = {}
    if out_file.is_file():
        for line in out_file.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("source") not in ("error", "no_summary"):
                done[r["match_key"]] = r
        print(f"resuming — {len(done)} already scored")

    client = None
    counts = {"llm": 0, "no_summary": 0, "error": 0, "skipped": 0}

    for row in metadata:
        key = row["match_key"]
        if key in done:
            counts["skipped"] += 1
            continue

        sf = _summary_for(row)
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
        with out_file.open("a", encoding="utf-8") as fh:  # append-as-we-go
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Rewrite a clean, de-duplicated file (one row per match_key).
    with out_file.open("w", encoding="utf-8") as fh:
        for r in done.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    scored = [r for r in done.values() if r.get("content_score") is not None]
    print(f"scored {len(scored)}/{len(metadata)} interview(s) -> {out_file}")
    print(f"  llm={counts['llm']} no_summary={counts['no_summary']} "
          f"error={counts['error']} skipped={counts['skipped']}")
    for r in sorted(scored, key=lambda x: -x["content_score"]):
        print(f"  {r['content_score']:.3f}  "
              f"spec={r['specificity']:.2f} exp={r['experiential_depth']:.2f} "
              f"ins={r['insight']:.2f}  {r['match_key']}")


if __name__ == "__main__":
    run()

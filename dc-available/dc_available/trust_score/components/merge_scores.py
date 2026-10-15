"""Merge the metadata and content-score tables into one row per document.

Deterministic join on match_key:

    metadata_aligned.jsonl  (freshness/authority/organisation raw fields)
        +  content_scores.jsonl  (content_score + sub-scores)
        ->  metadata_scored.jsonl

This is the single input the Stage-B prior scorer will read. No scoring here —
just a join, so re-running the LLM step and re-merging is cheap and safe.

RUN (from the dc-available directory):
    python -m dc_available.trust_score.prior_score
"""
from __future__ import annotations

import json
from pathlib import Path

_TRUST = (
    Path(__file__).resolve().parent.parent
    / "data" / "Artifacts" / "resources" / "trust" / "grownfield"
)
_METADATA = _TRUST / "metadata_aligned.jsonl"
_CONTENT = _TRUST / "content_scores.jsonl"
_OUT = _TRUST / "metadata_scored.jsonl"

_SUB_KEYS = ["technical_depth", "process_completeness", "training_usefulness", "focus"]


def merge() -> None:
    metadata = [json.loads(line) for line in _METADATA.open(encoding="utf-8")]
    content = {
        json.loads(line)["match_key"]: json.loads(line)
        for line in _CONTENT.open(encoding="utf-8")
    }

    out: list[dict] = []
    missing = 0
    for row in metadata:
        c = content.get(row["match_key"])
        if c is None:
            missing += 1
            row["content_score"] = None
            row["content_source"] = "missing"
        else:
            row["content_score"] = c["content_score"]
            row["content_source"] = c["source"]           # llm | video_default
            row["content_reason"] = c.get("reason", "")
            for k in _SUB_KEYS:
                row[k] = c.get(k)
        out.append(row)

    with _OUT.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"merged rows          : {len(out)}  -> {_OUT.name}")
    print(f"content missing      : {missing}")
    print(f"with content_score   : {len(out) - missing}")


if __name__ == "__main__":
    merge()

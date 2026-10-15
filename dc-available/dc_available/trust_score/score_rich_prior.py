"""Stage B (RICH prior) — merge content + metadata into one 0-1 trust prior per interview.

RICH analogue of ``components/merge_scores.py`` + ``components/score_prior.py`` (both
grownfield-pinned), combined into one small step. It REUSES ``score_prior``'s pillar functions
and WEIGHTS unchanged, so the 4-pillar formula is identical to QA:

    prior = 0.45*content + 0.25*authority + 0.15*format + 0.15*freshness

The four pillars are independent; this just combines them. (For this RICH upload authority and
freshness happen to be constant, but nothing here assumes that — the same code works if a
future dataset varies them.)

Join: ``content_scores.jsonl`` is joined to ``metadata.jsonl`` by ``match_key``.

Input : Artifacts/rich/trust/{metadata.jsonl, content_scores.jsonl}
Output: Artifacts/rich/trust/metadata_prior.jsonl

RUN (from the dc-available directory):
    python -m dc_available.trust_score.score_rich_prior
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from dc_available.config import data_config
from dc_available.trust_score.components.score_prior import (
    WEIGHTS,
    authority_score,
    format_score,
    freshness_score,
)


def _trust_dir() -> Path:
    return data_config.source_artifact_paths("rich")["ingestion"].parent / "trust"


def run() -> None:
    trust_dir = _trust_dir()
    metadata = [json.loads(line) for line in (trust_dir / "metadata.jsonl").open(encoding="utf-8")]

    content: dict[str, dict] = {}
    for line in (trust_dir / "content_scores.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        content[r["match_key"]] = r

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    for m in metadata:
        key = m["match_key"]
        c = content.get(key, {})

        cs = c.get("content_score")
        cs = 1.0 if cs is None else float(cs)  # safety fallback (mirrors score_prior)
        auth = authority_score(m.get("created_by", ""), m.get("modified_by", ""))
        fmt = format_score(m.get("file_type", ""))
        fresh = freshness_score(m.get("modified", ""), now)

        row = dict(m)  # carry raw metadata through
        # content pillar + its sub-scores (transparency / later audit)
        row["content_score"] = c.get("content_score")
        row["specificity"] = c.get("specificity")
        row["experiential_depth"] = c.get("experiential_depth")
        row["insight"] = c.get("insight")
        row["content_reason"] = c.get("reason")
        # the other three pillars
        row["authority_score"] = round(auth, 4)
        row["format_score"] = round(fmt, 4)
        row["freshness_score"] = round(fresh, 4)
        row["prior"] = round(
            WEIGHTS["content"] * cs
            + WEIGHTS["authority"] * auth
            + WEIGHTS["format"] * fmt
            + WEIGHTS["freshness"] * fresh,
            4,
        )
        rows.append(row)

    out_file = trust_dir / "metadata_prior.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    priors = [r["prior"] for r in rows]
    print(f"scored {len(rows)} interview(s) -> {out_file}")
    print(f"prior  min/mean/max/std : "
          f"{min(priors):.3f} / {mean(priors):.3f} / {max(priors):.3f} / {pstdev(priors):.3f}")
    for r in sorted(rows, key=lambda x: -x["prior"]):
        print(f"  prior={r['prior']:.3f}  "
              f"content={r.get('content_score')}  auth={r['authority_score']:.2f}  "
              f"fmt={r['format_score']:.2f}  fresh={r['freshness_score']:.2f}  {r['match_key']}")


if __name__ == "__main__":
    run()

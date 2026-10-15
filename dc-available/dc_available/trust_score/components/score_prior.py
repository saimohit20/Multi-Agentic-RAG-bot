"""Stage B — the metadata PRIOR: one 0-1 trust prior per document.

Combines four pillars into a single query-independent prior:

    prior = w_content·content + w_authority·authority
          + w_format·format   + w_freshness·freshness

Each pillar is a 0-1 sub-score; the weights sum to 1 so the prior lands in 0-1.
Weights/maps live at the top so they can be tuned against the human gold set
(Test 1) later without touching the logic.

Design notes (grounded in this corpus):
    * content   — the only strong discriminator (0.30-1.00).
    * authority — internal vs EXTERNAL vendor is the real org signal (in the
                  created_by string); topic is NOT used (it's the domain).
    * format    — user weighting: video/pptx high, pdf/url lower.
    * freshness — all docs are within ~1 year, so a gentle exponential decay is
                  used (yearly buckets would make it constant). Small weight.

Input : metadata_scored.jsonl   (metadata + content_score)
Output: metadata_prior.jsonl    (+ the 4 sub-scores + prior)

RUN (from the dc-available directory):
    python -m dc_available.trust_score.prior_score
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

# --------------------------------------------------------------------------- #
# Tunable configuration
# --------------------------------------------------------------------------- #
WEIGHTS = {
    "content": 0.45,
    "authority": 0.25,
    "format": 0.15,
    "freshness": 0.15,
}

AUTHORITY = {"internal": 1.0, "external": 0.7, "unknown": 0.3}

FORMAT_SCORE = {
    "mp4": 1.0, "srt": 1.0,   # video + transcript
    "pptx": 1.0,
    "docx": 0.7,
    "pdf": 0.6,
    "url": 0.5,
}
FORMAT_DEFAULT = 0.6

FRESHNESS_HALF_LIFE_YEARS = 3.0   # gentle: recency != trust
FRESHNESS_FLOOR = 0.3
FRESHNESS_UNDATED = 0.4

_TRUST = (
    Path(__file__).resolve().parent.parent
    / "data" / "Artifacts" / "resources" / "trust" / "grownfield"
)
_IN = _TRUST / "metadata_scored.jsonl"
_OUT = _TRUST / "metadata_prior.jsonl"


# --------------------------------------------------------------------------- #
# Pillar sub-scores
# --------------------------------------------------------------------------- #
def authority_score(created_by: str, modified_by: str) -> float:
    s = (created_by or modified_by or "").strip()
    if not s:
        return AUTHORITY["unknown"]
    if "EXTERNAL" in s.upper():
        return AUTHORITY["external"]
    return AUTHORITY["internal"]


def format_score(file_type: str) -> float:
    return FORMAT_SCORE.get((file_type or "").lower(), FORMAT_DEFAULT)


def freshness_score(modified_iso: str, now: datetime) -> float:
    if not modified_iso:
        return FRESHNESS_UNDATED
    try:
        dt = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
    except ValueError:
        return FRESHNESS_UNDATED
    age_years = max(0.0, (now - dt).days / 365.25)
    score = 0.5 ** (age_years / FRESHNESS_HALF_LIFE_YEARS)
    return max(FRESHNESS_FLOOR, min(1.0, score))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run() -> None:
    rows = [json.loads(line) for line in _IN.open(encoding="utf-8")]
    now = datetime.now(timezone.utc)

    for r in rows:
        content = r.get("content_score")
        content = 1.0 if content is None else float(content)  # safety fallback
        auth = authority_score(r.get("created_by", ""), r.get("modified_by", ""))
        fmt = format_score(r.get("file_type", ""))
        fresh = freshness_score(r.get("modified", ""), now)

        r["authority_score"] = round(auth, 4)
        r["format_score"] = round(fmt, 4)
        r["freshness_score"] = round(fresh, 4)
        r["prior"] = round(
            WEIGHTS["content"] * content
            + WEIGHTS["authority"] * auth
            + WEIGHTS["format"] * fmt
            + WEIGHTS["freshness"] * fresh,
            4,
        )

    with _OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    priors = [r["prior"] for r in rows]
    print(f"scored {len(rows)} docs -> {_OUT.name}")
    print(f"prior  min/mean/max/std : "
          f"{min(priors):.3f} / {mean(priors):.3f} / {max(priors):.3f} / {pstdev(priors):.3f}")
    print("\nlowest 5 priors (least trusted):")
    for r in sorted(rows, key=lambda x: x["prior"])[:5]:
        print(f"  {r['prior']:.3f}  {r.get('file_type',''):5s}  "
              f"c={r.get('content_score')}  {r['match_key'][:50]}")
    print("\nhighest 5 priors:")
    for r in sorted(rows, key=lambda x: x["prior"], reverse=True)[:5]:
        print(f"  {r['prior']:.3f}  {r.get('file_type',''):5s}  "
              f"c={r.get('content_score')}  {r['match_key'][:50]}")


if __name__ == "__main__":
    run()

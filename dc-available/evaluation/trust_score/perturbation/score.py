"""Step 3 — score every variant with the REAL trust-score code (prior only).

For each manifest row we recompute the four content sub-scores, the three
metadata pillars, and the ``prior`` — reusing the exact production functions so
results reflect the real pipeline:

    * content perturbations -> re-run score_content._score_summary() (gpt-4o) on
      the perturbed summary; metadata pillars copied from the baseline.
    * originals / metadata  -> content sub-scores copied from the baseline; the
      one overridden field (authority or freshness) recomputed via score_prior.

All pillars are recomputed with a single ``now`` so the baseline and perturbed
rows in results.jsonl are internally consistent. TrustRank/corroboration is NOT
touched — ``final_trust`` is out of scope for this experiment.

Input : perturbation/sample.jsonl + perturbation/perturbations.jsonl
Output: perturbation/results.jsonl  (240 rows: pillars + prior)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.perturbation.score
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from dc_available.config.llm_config import get_openai_client
from dc_available.trust_score.components.score_content import _SUB_KEYS, _read_text, _score_summary
from dc_available.trust_score.components.score_prior import (
    WEIGHTS,
    authority_score,
    format_score,
    freshness_score,
)
from evaluation.trust_score.perturbation.config import (
    CONTENT_TYPES,
    MANIFEST_FILE,
    PERT_DIR,
    RESULTS_FILE,
    SAMPLE_FILE,
)


def _prior(content: float, auth: float, fmt: float, fresh: float) -> float:
    return round(
        WEIGHTS["content"] * content
        + WEIGHTS["authority"] * auth
        + WEIGHTS["format"] * fmt
        + WEIGHTS["freshness"] * fresh,
        4,
    )


def run(resume: bool = True) -> None:
    base = {d["doc_id"]: d
            for d in (json.loads(line) for line in SAMPLE_FILE.open(encoding="utf-8"))}
    manifest = [json.loads(line) for line in MANIFEST_FILE.open(encoding="utf-8")]
    now = datetime.now(timezone.utc)

    done: dict[str, dict] = {}
    if resume and RESULTS_FILE.is_file():
        for line in RESULTS_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
        print(f"resuming — {len(done)} already scored")

    client = None
    scored = 0
    for m in manifest:
        if m["id"] in done:
            continue
        b = base[m["base"]]
        pert = m["perturbation"]

        # --- content sub-scores ---
        if pert in CONTENT_TYPES:
            text = _read_text(PERT_DIR / m["input"]["summary_file"])
            if client is None:
                client = get_openai_client()
            subs = _score_summary(client, text)
            sub_out = {k: subs[k] for k in _SUB_KEYS}
            content_score = subs["content_score"]
        else:  # original or metadata -> baseline content, unchanged
            sub_out = {k: b.get(k, b.get("content_score", 1.0)) for k in _SUB_KEYS}
            content_score = b.get("content_score", 1.0)

        # --- metadata pillars (apply any override) ---
        ovr = m["input"]["override"] if isinstance(m["input"], dict) and "override" in m["input"] else {}
        created_by = ovr.get("created_by", b.get("created_by", ""))
        modified_by = ovr.get("modified_by", b.get("modified_by", ""))
        modified = ovr.get("modified", b.get("modified", ""))

        auth = authority_score(created_by, modified_by)
        fmt = format_score(b.get("file_type", ""))
        fresh = freshness_score(modified, now)

        row = {
            "id": m["id"], "base": m["base"], "perturbation": pert,
            **{k: round(float(sub_out[k]), 4) for k in _SUB_KEYS},
            "content_score": round(float(content_score), 4),
            "authority_score": round(auth, 4),
            "format_score": round(fmt, 4),
            "freshness_score": round(fresh, 4),
            "prior": _prior(float(content_score), auth, fmt, fresh),
        }
        done[m["id"]] = row
        scored += 1
        with RESULTS_FILE.open("a", encoding="utf-8") as fh:   # crash-safe append
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if scored % 20 == 0:
            print(f"  ...{scored} scored")

    # rewrite clean, de-duplicated, in manifest order
    order = {m["id"]: i for i, m in enumerate(manifest)}
    with RESULTS_FILE.open("w", encoding="utf-8") as fh:
        for r in sorted(done.values(), key=lambda x: order.get(x["id"], 1 << 30)):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"scored {scored} new / {len(done)} total -> {RESULTS_FILE.name}")


if __name__ == "__main__":
    run()

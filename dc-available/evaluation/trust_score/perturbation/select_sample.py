
from __future__ import annotations

import json
import random
from collections import defaultdict

from dc_available.trust_score.components.score_content import _summary_index
from evaluation.trust_score.perturbation.config import (
    BASELINE_FILE,
    N_DOCS,
    PERT_DIR,
    SAMPLE_FILE,
    SEED,
)


def run(n_docs: int = N_DOCS) -> None:
    rows = [json.loads(line) for line in BASELINE_FILE.open(encoding="utf-8")]
    summaries = _summary_index()

    # candidates: llm-scored text docs with a resolvable summary
    cands = []
    for r in rows:
        if r.get("content_source") != "llm":
            continue
        sp = summaries.get(r["match_key"])
        if sp is None:
            continue
        r = dict(r)
        r["summary_path"] = str(sp)
        cands.append(r)

    # stratified round-robin across topics (even spread, seeded)
    by_topic: dict[str, list] = defaultdict(list)
    for r in cands:
        by_topic[r.get("topic", "?")].append(r)
    rnd = random.Random(SEED)
    for lst in by_topic.values():
        rnd.shuffle(lst)

    topics = sorted(by_topic)
    picked: list = []
    i = 0
    while len(picked) < n_docs and any(by_topic[t] for t in topics):
        t = topics[i % len(topics)]
        if by_topic[t]:
            picked.append(by_topic[t].pop())
        i += 1
    picked = picked[:n_docs]

    for n, r in enumerate(picked, 1):
        r["doc_id"] = f"doc{n:03d}"

    PERT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE_FILE.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_topic = defaultdict(int)
    for r in picked:
        per_topic[r.get("topic", "?")] += 1
    print(f"candidates : {len(cands)} llm-scored text docs")
    print(f"selected   : {len(picked)} docs -> {SAMPLE_FILE.name}")
    print("per topic  : " + ", ".join(f"{t}={n}" for t, n in sorted(per_topic.items())))
    priors = [r["prior"] for r in picked]
    print(f"prior range: {min(priors):.3f} .. {max(priors):.3f}")


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_DOCS)
